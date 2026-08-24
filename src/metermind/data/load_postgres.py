"""Apply the Postgres schema and load the site register.

The document corpus for RAG arrives in Week 2; this module currently only fills
the `sites` table, which is enough to prove the pgvector container is real and
reachable rather than an empty service in docker-compose.yml.
"""

from __future__ import annotations

import psycopg

from metermind.config import SQL_DIR, PostgresConfig, postgres_config

SCHEMA_FILE = SQL_DIR / "schema.sql"

SITE_COLUMNS = (
    "site_id",
    "name",
    "category",
    "city",
    "meter_id",
    "floor_area_m2",
    "base_load_kw",
    "peak_load_kw",
    "commissioned_on",
    "notes",
)


def check_connection(cfg: PostgresConfig | None = None) -> None:
    cfg = cfg or postgres_config()
    try:
        with psycopg.connect(cfg.dsn, connect_timeout=10) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001 - surface the cause verbatim
        raise RuntimeError(
            f"Cannot reach Postgres at {cfg.host}:{cfg.port} ({exc}).\n"
            f"  Is the stack up?   docker compose up -d\n"
            f"  Is it healthy?     docker compose ps"
        ) from exc


def apply_schema(cfg: PostgresConfig | None = None) -> None:
    """Run docker/postgres/schema.sql.

    Every statement in that file is idempotent, so this runs on each build.
    Keeping tables here rather than in docker-entrypoint-initdb.d means adding a
    table in Week 2 will not require destroying the volume -- initdb scripts run
    exactly once, on an empty data directory, and never again.
    """
    cfg = cfg or postgres_config()
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    with psycopg.connect(cfg.dsn) as conn:
        conn.execute(sql)
        conn.commit()


def upsert_sites(sites: list[dict], cfg: PostgresConfig | None = None) -> int:
    """Insert or update the site register from sites.yaml."""
    cfg = cfg or postgres_config()

    columns = ", ".join(SITE_COLUMNS)
    placeholders = ", ".join(f"%({column})s" for column in SITE_COLUMNS)
    updates = ", ".join(f"{column} = EXCLUDED.{column}" for column in SITE_COLUMNS if column != "site_id")
    statement = (
        f"INSERT INTO sites ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT (site_id) DO UPDATE SET {updates}"
    )

    rows = [{column: site.get(column) for column in SITE_COLUMNS} for site in sites]
    with psycopg.connect(cfg.dsn) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(statement, rows)
        conn.commit()
    return len(rows)
