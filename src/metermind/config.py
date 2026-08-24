"""Central configuration, read from .env.

docker-compose.yml reads the same .env file, so the containers and this code
cannot disagree about credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# src/metermind/config.py -> src/metermind -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GENERATED_DIR = DATA_DIR / "generated"
EVAL_DIR = PROJECT_ROOT / "eval"
SQL_DIR = PROJECT_ROOT / "docker" / "postgres"

SITES_FILE = GENERATED_DIR / "sites.yaml"
ANOMALIES_FILE = GENERATED_DIR / "anomalies.json"
BUILD_MANIFEST_FILE = GENERATED_DIR / "build_manifest.json"

# InfluxDB measurement names. Real and synthetic data are kept in separate
# measurements so the distinction is enforced by the schema rather than by a
# comment somebody forgets to read.
MEASUREMENT_GRID = "grid_load"          # real, measured: OPSD German national load
MEASUREMENT_SITES = "site_consumption"  # synthetic: the eight fictional sites

# All local-time reasoning in this project uses this zone. Storage is always UTC.
LOCAL_TZ = "Europe/Berlin"


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in:\n"
            f"    cp .env.example .env"
        )
    return value


@dataclass(frozen=True)
class InfluxConfig:
    url: str
    token: str
    org: str
    bucket: str


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class BuildConfig:
    """Everything that determines the shape of the generated dataset.

    Changing any of these changes the numbers, which invalidates cached eval
    gold answers -- hence they are recorded in build_manifest.json.
    """

    seed: int
    interval_minutes: int
    history_days: int


def influx_config() -> InfluxConfig:
    return InfluxConfig(
        url=os.getenv("INFLUX_URL", "http://localhost:8086"),
        token=_require("INFLUX_TOKEN"),
        org=os.getenv("INFLUX_ORG", "metermind"),
        bucket=os.getenv("INFLUX_BUCKET", "energy"),
    )


def postgres_config() -> PostgresConfig:
    return PostgresConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=_require("POSTGRES_USER"),
        password=_require("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB", "metermind"),
    )


def build_config() -> BuildConfig:
    return BuildConfig(
        seed=int(os.getenv("METERMIND_SEED", "20260814")),
        interval_minutes=int(os.getenv("METERMIND_INTERVAL_MINUTES", "15")),
        history_days=int(os.getenv("METERMIND_HISTORY_DAYS", "548")),
    )
