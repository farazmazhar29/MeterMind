#!/usr/bin/env python
"""Build the MeterMind dataset end to end.

    python scripts/build_dataset.py            # full build
    python scripts/build_dataset.py --dry-run  # generate, print stats, write nothing
    python scripts/build_dataset.py --only-real

Stages:
  1. Download the real OPSD German load series (cached after the first run).
  2. Derive a seasonal envelope from it and time-shift it to the present.
  3. Generate eight synthetic sites from that envelope.       [needs synth.py]
  4. Inject anomalies and record their ground truth.          [needs anomalies.py]
  5. Write everything to InfluxDB and the site register to Postgres.

Stages 3 and 4 need the modelling modules described in docs/CONTRACTS.md. Until
they exist this script runs stages 1, 2 and 5-for-real-data only, and says
exactly what is missing. That is intentional: the real half of the pipeline is
verifiable today rather than blocked behind unwritten code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from metermind import config
from metermind.config import (
    ANOMALIES_FILE,
    BUILD_MANIFEST_FILE,
    GENERATED_DIR,
    MEASUREMENT_GRID,
    MEASUREMENT_SITES,
    SITES_FILE,
    build_config,
)
from metermind.data import load_influx, load_postgres, opsd

# Modules Faraz is writing, and the callable each must expose.
# See docs/CONTRACTS.md for the full signatures.
MODELLING_MODULES = {
    "metermind.data.profiles": "category_shape",
    "metermind.data.synth": "generate_site_series",
    "metermind.data.anomalies": "inject",
}


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def load_sites() -> list[dict]:
    with SITES_FILE.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["sites"]


def resolve_modelling_modules() -> tuple[dict | None, list[str]]:
    """Import the modelling modules, or report precisely what is missing."""
    resolved: dict = {}
    problems: list[str] = []

    for module_name, required_fn in MODELLING_MODULES.items():
        short = module_name.rsplit(".", 1)[-1]
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            problems.append(f"  missing  src/metermind/data/{short}.py")
            continue
        if not hasattr(module, required_fn):
            problems.append(f"  missing  {short}.{required_fn}()")
            continue
        resolved[short] = module

    return (resolved if not problems else None), problems


def target_index(interval_minutes: int, history_days: int) -> pd.DatetimeIndex:
    """Timestamps for the synthetic sites: `history_days` back from now."""
    now = pd.Timestamp.now(tz="UTC").floor(f"{interval_minutes}min")
    start = now - pd.Timedelta(days=history_days)
    return pd.date_range(start=start, end=now, freq=f"{interval_minutes}min", tz="UTC")


def build_site_frame(series_by_site: dict[str, pd.Series], sites: list[dict]) -> pd.DataFrame:
    """Long-format frame ready for InfluxDB: index + tag columns + field column."""
    category_of = {site["site_id"]: site["category"] for site in sites}
    parts = []
    for site_id, series in series_by_site.items():
        part = pd.DataFrame(
            {
                "site_id": site_id,
                "category": category_of[site_id],
                "power_kw": series.astype("float64").to_numpy(),
            },
            index=series.index,
        )
        parts.append(part)
    return pd.concat(parts).sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="generate but write nothing")
    parser.add_argument("--only-real", action="store_true", help="skip the synthetic sites")
    parser.add_argument("--force-download", action="store_true", help="re-download the OPSD CSV")
    parser.add_argument("--skip-postgres", action="store_true")
    parser.add_argument("--skip-influx", action="store_true")
    args = parser.parse_args()

    cfg = build_config()
    rng = np.random.default_rng(cfg.seed)
    built_at = datetime.now(timezone.utc)

    # -- 1. Real data ------------------------------------------------------
    rule("1. Real data: Open Power System Data")
    csv_path = opsd.download(force=args.force_download)
    grid_load = opsd.load_german_load(csv_path)
    first, last = opsd.coverage(grid_load)
    print(f"  {opsd.LOAD_COLUMN}")
    print(f"  {len(grid_load):,} hourly points, {first:%Y-%m-%d} to {last:%Y-%m-%d}")
    print(f"  mean {grid_load.mean():,.0f} MW   peak {grid_load.max():,.0f} MW")

    # -- 2. Time offset and envelope --------------------------------------
    rule("2. Seasonal envelope")
    offset = opsd.compute_time_offset(last, built_at)
    print(f"  offset {offset.days} days ({offset.days / 364:.0f} x 364d)")
    print(f"  real series ends {last:%Y-%m-%d (%a)} -> shifted {last + offset:%Y-%m-%d (%a)}")
    if offset.days % 7 != 0:
        raise AssertionError("offset must be a whole number of weeks")

    index = target_index(cfg.interval_minutes, cfg.history_days)
    envelope = opsd.seasonal_envelope(grid_load, index, offset)
    print(f"  window {index[0]:%Y-%m-%d} to {index[-1]:%Y-%m-%d}")
    print(f"  {len(index):,} timestamps at {cfg.interval_minutes} min")
    print(f"  envelope range {envelope.min():.3f} to {envelope.max():.3f} (centred on 1.0)")

    sites = load_sites()
    print(f"  {len(sites)} sites in the register")

    # -- 3 & 4. Synthetic sites and anomalies ------------------------------
    series_by_site: dict[str, pd.Series] = {}
    anomaly_records: list[dict] = []
    modules, problems = (None, []) if args.only_real else resolve_modelling_modules()

    if args.only_real:
        rule("3. Synthetic sites: skipped (--only-real)")
    elif modules is None:
        rule("3. Synthetic sites: SKIPPED")
        print("  The modelling modules are not ready yet:")
        print("\n".join(problems))
        print("\n  See docs/CONTRACTS.md for the signatures they must match.")
        print("  Stages 1, 2 and the real-data load below still run.")
    else:
        rule("3. Synthetic sites")
        for site in sites:
            series = modules["synth"].generate_site_series(
                site=site, envelope=envelope, index=index, rng=rng
            )
            if not series.index.equals(index):
                raise ValueError(f"{site['site_id']}: returned index does not match the target index")
            if series.isna().any():
                raise ValueError(f"{site['site_id']}: series contains NaN")
            series_by_site[site["site_id"]] = series
            print(
                f"  {site['site_id']:<22} mean {series.mean():7.1f} kW   "
                f"peak {series.max():7.1f} kW   min {series.min():6.1f} kW"
            )

        rule("4. Anomaly injection")
        series_by_site, anomaly_records = modules["anomalies"].inject(
            series_by_site=series_by_site, sites=sites, rng=rng
        )
        by_type: dict[str, int] = {}
        for record in anomaly_records:
            by_type[record["type"]] = by_type.get(record["type"], 0) + 1
        print(f"  {len(anomaly_records)} anomalies: " + ", ".join(f"{n} {t}" for t, n in sorted(by_type.items())))
        missing_cause = [r["anomaly_id"] for r in anomaly_records if not r.get("cause")]
        if missing_cause:
            # Week 2's RAG corpus is generated from these strings. Catch it now,
            # while regenerating is still cheap.
            print(f"  WARNING: no `cause` on {len(missing_cause)} anomalies: {missing_cause}")

        if not args.dry_run:
            ANOMALIES_FILE.write_text(json.dumps(anomaly_records, indent=2), encoding="utf-8")
            print(f"  wrote {ANOMALIES_FILE.relative_to(config.PROJECT_ROOT)}")

    # -- 5. Load ------------------------------------------------------------
    rule("5. Load")
    if args.dry_run:
        print("  --dry-run: nothing written")
    else:
        if not args.skip_influx:
            load_influx.check_connection()

            grid_frame = pd.DataFrame(
                {"source": "opsd", "load_mw": grid_load.astype("float64").to_numpy()},
                index=grid_load.index,
            )
            load_influx.delete_measurement(MEASUREMENT_GRID)
            load_influx.write_frame(grid_frame, MEASUREMENT_GRID, tag_columns=["source"])

            if series_by_site:
                site_frame = build_site_frame(series_by_site, sites)
                load_influx.delete_measurement(MEASUREMENT_SITES)
                load_influx.write_frame(
                    site_frame, MEASUREMENT_SITES, tag_columns=["site_id", "category"]
                )
        else:
            print("  InfluxDB: skipped")

        if not args.skip_postgres:
            load_postgres.check_connection()
            load_postgres.apply_schema()
            count = load_postgres.upsert_sites(sites)
            print(f"    sites: {count} rows upserted")
        else:
            print("  Postgres: skipped")

    # -- Manifest -----------------------------------------------------------
    manifest = {
        "built_at_utc": built_at.isoformat(),
        "seed": cfg.seed,
        "interval_minutes": cfg.interval_minutes,
        "history_days": cfg.history_days,
        "window_start_utc": index[0].isoformat(),
        "window_end_utc": index[-1].isoformat(),
        "time_offset_days": offset.days,
        "real_data": {
            "source": "Open Power System Data, time_series package",
            "version": opsd.OPSD_VERSION,
            "url": opsd.OPSD_URL,
            "column": opsd.LOAD_COLUMN,
            "coverage_start_utc": first.isoformat(),
            "coverage_end_utc": last.isoformat(),
            "measurement": MEASUREMENT_GRID,
            "note": "Loaded at its true timestamps. The offset above applies only to the synthetic sites.",
        },
        "synthetic": {
            "measurement": MEASUREMENT_SITES,
            "sites": [site["site_id"] for site in sites],
            "generated": bool(series_by_site),
            "anomaly_count": len(anomaly_records),
        },
    }
    if not args.dry_run:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        BUILD_MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\n  wrote {BUILD_MANIFEST_FILE.relative_to(config.PROJECT_ROOT)}")

    rule("Done")
    if not series_by_site and not args.only_real:
        print("  Real data only. Write the modules in docs/CONTRACTS.md, then re-run.")
        return 0
    print("  Dataset complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
