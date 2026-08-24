"""Real data: Open Power System Data, German national electricity load.

THIS IS THE ONLY REAL MEASURED DATA IN THE PROJECT. Everything under
`site_consumption` in InfluxDB is synthetic. See README.md, "Real vs synthetic".

Source:  https://data.open-power-system-data.org/time_series/2020-10-06/
Column:  DE_load_actual_entsoe_transparency (MW, hourly, from ENTSO-E Transparency)
Licence: MIT for the package; underlying data attributed to ENTSO-E / TSOs.

Note on staleness: the OPSD time-series package was last published on
2020-10-06 and covers 2015 to mid-2020. It is not updated any more. That is why
`compute_time_offset` exists -- see the docstring there.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from metermind.config import RAW_DIR

OPSD_VERSION = "2020-10-06"
OPSD_FILENAME = "time_series_60min_singleindex.csv"
OPSD_URL = f"https://data.open-power-system-data.org/time_series/{OPSD_VERSION}/{OPSD_FILENAME}"

# The German national load column. Hourly, in MW.
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"

# Always use the UTC column. `cet_cest_timestamp` carries the DST discontinuities
# -- a missing hour every March and a duplicated hour every October -- which turn
# into silent off-by-one-hour errors downstream.
TIME_COLUMN = "utc_timestamp"

# 364 days = exactly 52 weeks. See compute_time_offset.
CALENDAR_STEP = timedelta(days=364)


def download(dest_dir: Path | None = None, force: bool = False) -> Path:
    """Download the OPSD CSV (~124 MB) unless it is already cached.

    Idempotent: re-running the pipeline does not re-download.
    """
    dest_dir = dest_dir or RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / OPSD_FILENAME

    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  [cached] {dest.name} ({size_mb:.1f} MB)")
        return dest

    print(f"  downloading {OPSD_URL}")
    print("  (~124 MB, this takes a few minutes on a slow connection)")

    digest = hashlib.sha256()
    tmp = dest.with_suffix(".csv.partial")
    with requests.get(OPSD_URL, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        written = 0
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                print(f"\r  {written / 1024 / 1024:8.1f} MB", end="", flush=True)
    print()

    # Rename only after a complete download, so an interrupted run cannot leave
    # a truncated file that later looks like a valid cache hit.
    tmp.replace(dest)
    print(f"  sha256 {digest.hexdigest()[:16]}...")
    return dest


def load_german_load(path: Path | None = None) -> pd.Series:
    """Read the German national load series.

    Returns hourly MW, indexed by a tz-aware UTC DatetimeIndex, with gaps
    dropped. Only two columns are parsed out of the ~400 in the file, so peak
    memory stays around 20 MB rather than 1.5 GB.
    """
    path = path or (RAW_DIR / OPSD_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/build_dataset.py` which "
            f"downloads it, or call opsd.download() directly."
        )

    frame = pd.read_csv(
        path,
        usecols=[TIME_COLUMN, LOAD_COLUMN],
        parse_dates=[TIME_COLUMN],
    )
    series = frame.set_index(TIME_COLUMN)[LOAD_COLUMN]

    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")

    # ENTSO-E has genuine reporting gaps. Drop them rather than interpolating:
    # the envelope is a daily mean, so a handful of missing hours is harmless,
    # and inventing values in real data would undermine the point of using it.
    series = series.dropna().sort_index()
    series.name = "load_mw"
    return series


def compute_time_offset(
    source_end: datetime,
    target_end: datetime | None = None,
) -> timedelta:
    """Offset that moves the OPSD series forward to end at or after `target_end`.

    The OPSD package stops in mid-2020, so questions like "what happened last
    Tuesday" have no data to land on. The synthetic sites are therefore built on
    a time-shifted copy of the real series.

    The offset is always a whole multiple of 364 days (= 52 weeks) because that
    is the only step that satisfies both constraints at once:

      * 364 is divisible by 7, so every timestamp keeps its weekday. A Monday
        stays a Monday, which matters because the whole dataset is built on
        weekday/weekend behaviour.
      * 364 is within 1.25 days of a calendar year, so seasonality barely moves
        -- about 7.5 days of drift over six years. Shifting by whole *weeks*
        instead would preserve weekdays just as well but slide summer into
        autumn, which would be far more visible than a one-week drift.

    Rounds up, so the shifted series extends slightly past `target_end`; the
    caller truncates at "now". Returns a zero offset if no shift is needed.
    """
    target_end = target_end or datetime.now(timezone.utc)

    if source_end.tzinfo is None:
        source_end = source_end.replace(tzinfo=timezone.utc)
    if target_end.tzinfo is None:
        target_end = target_end.replace(tzinfo=timezone.utc)

    if target_end <= source_end:
        return timedelta(0)

    steps = math.ceil((target_end - source_end) / CALENDAR_STEP)
    return steps * CALENDAR_STEP


def seasonal_envelope(
    load: pd.Series,
    index: pd.DatetimeIndex,
    offset: timedelta,
) -> pd.Series:
    """Dimensionless seasonal multiplier for the synthetic sites.

    Deliberately DAILY resolution. The real national curve also contains a
    strong intra-day cycle peaking in the evening, but that is the shape of
    aggregate residential and commercial demand -- not of a factory running two
    shifts, or a cold store that never switches off. Passing the intra-day cycle
    through to every site would make all eight sites near-perfectly correlated
    and the dataset obviously synthetic.

    So this strips the intra-day component and keeps only what genuinely
    generalises across sites: the seasonal swing (higher in winter) plus the
    real, weather-driven day-to-day variation that no noise model reproduces
    convincingly.

    Intra-day and weekly shape are supplied by profiles.py. See docs/CONTRACTS.md.

    Returns a Series on `index`, centred on ~1.0.
    """
    shifted = load.copy()
    shifted.index = shifted.index + offset

    # Daily mean on UTC day boundaries. The 1-2 hour offset from local midnight
    # is irrelevant at seasonal resolution.
    daily = shifted.resample("D").mean().dropna()
    if daily.empty:
        raise ValueError("No OPSD data left after resampling -- check the offset.")

    daily = daily / daily.mean()

    # Anchor each daily value at midday, then interpolate. Forward-filling
    # instead would put an artificial step change at every midnight, which the
    # anomaly detector would happily flag.
    daily.index = daily.index + pd.Timedelta(hours=12)

    combined = daily.reindex(daily.index.union(index)).interpolate(method="time")
    envelope = combined.reindex(index)

    # Only the first/last half-day of the window can still be NaN.
    envelope = envelope.ffill().bfill()
    envelope.name = "seasonal_envelope"
    return envelope


def coverage(load: pd.Series) -> tuple[datetime, datetime]:
    """(first, last) timestamp of the real series, as UTC datetimes."""
    return load.index[0].to_pydatetime(), load.index[-1].to_pydatetime()
