"""Write time series into InfluxDB 2.7.

Why InfluxDB 2.7 and not 3.x: InfluxDB 3 Core, the free OSS tier, restricts
queries to roughly the last 72 hours of data. This project queries months of
history, so 3 Core is unusable here and 2.7 is the pragmatic choice. The
trade-off is that Flux is on a deprecation path upstream.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

from metermind.config import InfluxConfig, influx_config

# Rows per write request. 25k keeps each HTTP payload a few MB, which is well
# inside Influx's default limits and gives useful progress output.
CHUNK_ROWS = 25_000


def client(cfg: InfluxConfig | None = None) -> InfluxDBClient:
    cfg = cfg or influx_config()
    return InfluxDBClient(url=cfg.url, token=cfg.token, org=cfg.org, timeout=60_000)


def check_connection(cfg: InfluxConfig | None = None) -> None:
    """Fail early with an actionable message rather than deep inside a write."""
    cfg = cfg or influx_config()
    try:
        with client(cfg) as influx:
            if not influx.ping():
                raise RuntimeError("ping returned false")
    except Exception as exc:  # noqa: BLE001 - surface the cause verbatim
        raise RuntimeError(
            f"Cannot reach InfluxDB at {cfg.url} ({exc}).\n"
            f"  Is the stack up?   docker compose up -d\n"
            f"  Is it healthy?     docker compose ps\n"
            f"  Right token?       INFLUX_TOKEN in .env must match the value\n"
            f"                     InfluxDB was initialised with. If you changed\n"
            f"                     it after first start, run `docker compose down -v`."
        ) from exc


def delete_measurement(measurement: str, cfg: InfluxConfig | None = None) -> None:
    """Clear a measurement so a rebuild replaces rather than merges.

    Influx overwrites points with an identical (measurement, tag set, timestamp,
    field) key, so a same-seed rebuild is already idempotent. This matters when
    the shape changes -- a renamed site would otherwise leave orphan points
    behind that quietly pollute every later query.
    """
    cfg = cfg or influx_config()
    with client(cfg) as influx:
        influx.delete_api().delete(
            start=datetime(1970, 1, 1, tzinfo=timezone.utc),
            stop=datetime(2100, 1, 1, tzinfo=timezone.utc),
            predicate=f'_measurement="{measurement}"',
            bucket=cfg.bucket,
            org=cfg.org,
        )


def write_frame(
    frame: pd.DataFrame,
    measurement: str,
    tag_columns: list[str],
    cfg: InfluxConfig | None = None,
) -> int:
    """Write a DataFrame indexed by a tz-aware UTC DatetimeIndex.

    Columns named in `tag_columns` become tags; every other column becomes a
    field. Returns the number of rows written.
    """
    cfg = cfg or influx_config()

    if frame.empty:
        return 0
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{measurement}: index must be a DatetimeIndex")
    if frame.index.tz is None:
        raise TypeError(
            f"{measurement}: index must be tz-aware UTC. A naive index is "
            f"interpreted as local time and silently shifts every point."
        )

    missing = [column for column in tag_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{measurement}: tag columns not in frame: {missing}")

    total = 0
    with client(cfg) as influx:
        with influx.write_api(write_options=SYNCHRONOUS) as writer:
            for start in range(0, len(frame), CHUNK_ROWS):
                chunk = frame.iloc[start : start + CHUNK_ROWS]
                writer.write(
                    bucket=cfg.bucket,
                    record=chunk,
                    data_frame_measurement_name=measurement,
                    data_frame_tag_columns=tag_columns,
                )
                total += len(chunk)
                print(f"\r    {measurement}: {total:,} / {len(frame):,} rows", end="", flush=True)
    print()
    return total


def count_points(measurement: str, cfg: InfluxConfig | None = None) -> int:
    """Row count for a measurement. Used to verify a load actually landed."""
    cfg = cfg or influx_config()
    flux = f'''
from(bucket: "{cfg.bucket}")
  |> range(start: 1970-01-01T00:00:00Z, stop: 2100-01-01T00:00:00Z)
  |> filter(fn: (r) => r._measurement == "{measurement}")
  |> count()
  |> sum()
'''
    with client(cfg) as influx:
        tables = influx.query_api().query(flux)
    return sum(record.get_value() for table in tables for record in table.records)
