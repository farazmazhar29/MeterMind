"""Per-category load shapes: what happens inside a day and a week."""

from __future__ import annotations

import functools
from datetime import date

import holidays
import numpy as np
import pandas as pd

from metermind.config import LOCAL_TZ


WORKDAY = "workday"
SATURDAY = "saturday"
SUNDAY = "sunday"

SITE_STATE = {
    "nordwerk-kassel": "HE",      # Hessen
    "hansawerk-hamburg": "HH",    # Hamburg
    "sudpark-muenchen": "BY",     # Bayern
    "techcampus-dresden": "SN",   # Sachsen
    "frostlager-bremen": "HB",    # Bremen
    "kuehlhaus-duisburg": "NW",   # Nordrhein-Westfalen
    "marktplatz-koeln": "NW",     # Nordrhein-Westfalen
    "stadtmarkt-leipzig": "SN",   # Sachsen
}

SEASONAL_SENSITIVITY = {
    "manufacturing": 0.20,
    "office": -0.30,
    "cold_storage": -0.75,
    "retail": -0.25,
}

def _window(hours: np.ndarray, start: float, end: float, sharpness: float = 2.0) -> np.ndarray:
    """A smooth 0 -> 1 -> 0 window between `start` and `end` (decimal local hours)."""
    rise = 1.0 / (1.0 + np.exp(-sharpness * (hours - start)))
    fall = 1.0 / (1.0 + np.exp(-sharpness * (end - hours)))
    return rise * fall

@functools.lru_cache(maxsize=None)
def _holiday_calender(state: str | None, first_year: int, last_year: int) -> frozenset[date]:
    """Public holiays for one federal state. cached: called once per site."""
    years = range(first_year, last_year + 1)
    calender = holidays.Germany(years=years, subdiv=state)
    return frozenset(calender.keys())

def _day_type(local: pd.DatetimeIndex, site: dict) -> np.ndarray:
    """Classify every timestamp as workday / saturday / sunday."""
    state = SITE_STATE.get(site.get("site_id"))
    holiday_dates = _holiday_calender(state, int(local.year.min()), int(local.year.max()))

    day_type = np.full(len(local), WORKDAY, dtype=object)
    weekday = local.dayofweek.to_numpy()
    day_type[weekday == 5] = SATURDAY
    day_type[weekday == 6] = SUNDAY

    is_holiday = np.fromiter((d in holiday_dates for d in local.date), dtype=bool, count=len(local))
    day_type[is_holiday] = SUNDAY
    return day_type

def _manufacturing(hours:np.ndarray, day_type: np.ndarray, site: dict) -> np.ndarray:
    """Two weekday shifts, saturday maintenance, dead on Sunday."""
    standby = 0.06      # compressed air, lighting, controls
    shifts = _window(hours, 5.5, 22.0, 1.1)     # slow ramp: oven heats up
    changeouver = 0.18 * _window(hours, 13.5, 14.5, 6.0)    # shift handover lull

    weekday = standby + shifts - changeouver
    saturday = standby + 0.40 * _window(hours, 6.0, 12.5, 2.0)
    sunday = np.full(hours.shape, standby)
    return np.select([day_type == WORKDAY, day_type == SATURDAY], [weekday, saturday], sunday)

def _office(hours: np.ndarray, day_type: np.ndarray, site: dict) -> np.ndarray:
    """Occupied 07:00-19:00 on weekdays, near-empty otherwise."""
    overnight = 0.10                                 # servers, standby, emergency lighting
    occupied = _window(hours, 7.0, 19.0, 2.2)
    lunch = 0.08 * _window(hours, 12.0, 13.5, 5.0)

    weekday = overnight + 0.90 * occupied - lunch
    saturday = overnight + 0.14 * _window(hours, 9.0, 16.0, 2.0)
    sunday = np.full(hours.shape, overnight)

    return np.select([day_type == WORKDAY, day_type == SATURDAY], [weekday, saturday], sunday)


def _cold_storage(hours: np.ndarray, day_type: np.ndarray, site: dict) -> np.ndarray:
    """Never switches off. Driven by ambient temperature, not by the clock."""
    thermal = 0.62 + 0.20 * _window(hours, 10.0, 19.0, 1.0)    # afternoon heat load

    weekday = thermal + 0.10 * _window(hours, 6.0, 18.0, 2.0)  # door openings, traffic
    saturday = thermal + 0.05 * _window(hours, 7.0, 13.0, 2.0)
    sunday = thermal

    return np.select([day_type == WORKDAY, day_type == SATURDAY], [weekday, saturday], sunday)


def _retail(hours: np.ndarray, day_type: np.ndarray, site: dict) -> np.ndarray:
    """Trading hours, Saturday peak, CLOSED SUNDAY."""
    cases = 0.34                                     # refrigerated display, 24/7
    prep = 0.10 * _window(hours, 4.0, 8.0, 3.0)      # bakery, restocking
    trading = _window(hours, 7.0, 21.0, 3.0)

    weekday = cases + prep + 0.55 * trading
    saturday = cases + prep + 0.66 * trading         # busiest trading day
    sunday = np.full(hours.shape, cases)

    return np.select([day_type == WORKDAY, day_type == SATURDAY], [weekday, saturday], sunday)

_SHAPES = {
    "manufacturing": _manufacturing,
    "office": _office,
    "cold_storage": _cold_storage,
    "retail": _retail,
}

def category_shape(category: str, index: pd.DatetimeIndex, site: dict) -> pd.Series:
    """0.0 = at base load, 1.0 = at nameplate peak."""
    if category not in _SHAPES:
        raise KeyError(f"Unknown category {category!r}. Must be one of {sorted(_SHAPES)}.")
    
    local = index.tz_convert(LOCAL_TZ)
    hours = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    day_type = _day_type(local, site)

    shape = _SHAPES[category](hours, day_type, site)
    return pd.Series(np.clip(shape, 0.0, 1.0), index=index, name="shape")

def seasonal_sensitivity(category: str) -> float:
    """How strongly, and in which direction, a category tracks the OPSD envelope.

    Used by synth.py as:   effect = 1 + sensitivity * (envelope - 1)
    """
    if category not in SEASONAL_SENSITIVITY:
        raise KeyError(f"unknown category {category!r}")
    return SEASONAL_SENSITIVITY[category]

def holiday_dates(site: dict, index: pd.DatetimeIndex) -> frozenset[date]:
    """Public accessor for the holiday calendar that applies to one site.

    anomalies.py needs this to avoid placing short anomalies on public
    holidays, and reaching into a private function from another module is
    poor form.
    """
    state = SITE_STATE.get(site.get("site_id"))
    return _holiday_calender(state, int(index.year.min()), int(index.year.max()))
