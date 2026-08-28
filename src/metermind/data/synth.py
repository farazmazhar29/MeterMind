"""Turn a 0..1 category shape into kilowatts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metermind.data import profiles

# Sub-metering hardware error plus switching transients. small and fast.
MEASUREMENT_NOISE = 0.015

# What fraction of the base->peak span a site uses when its shape says "flat out"
# under *average* seasonal and activity conditions. Below 1.0 on purpose:
# nameplate peak is a design rating reached when demand and season align, not a
# number an ordinary Tuesday hits.
NOMINAL_UTILISATION = 0.82

# Business-volume drift: order books and occupancy vary week to week.
ACTIVITY_VOLATILITY = 0.035
ACTIVITY_REVERSION = 0.15

# Nameplate ratings are design figures, not hard limits. A brief excursion
# above peak is realistic; a sustained one is not.
MAX_OVERSHOOT = 1.15
MIN_UNDERSHOOT = 0.80

def _daily_activity(index: pd.DatetimeIndex, rng:np.random.Generator) -> pd.Series:
    """A slow, mean-reverting wander around 1.0. Some weeks are busier."""
    days = pd.date_range(index[0].floor("D"), index[-1].ceil("D"), freq="D")

    steps = rng.normal(0.0, ACTIVITY_VOLATILITY, size=len(days))
    deviation = np.zeros(len(days))
    for i in range(1, len(days)):
        deviation[i] = deviation[i-1] * (1.0 - ACTIVITY_REVERSION) + steps[i]

    daily = pd.Series(1.0 + deviation, index=days)
    daily.index = daily.index + pd.Timedelta(hours=12)  # center on noon
    combined = daily.reindex(daily.index.union(index)).interpolate(method="time")
    return combined.reindex(index).ffill().bfill().to_numpy()

def generate_site_series(site: dict, envelope: pd.Series, index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    """Return power in kW, indexed by `index`."""
    category = site["category"]
    base = float(site["base_load_kw"])
    peak = float(site["peak_load_kw"])
    span = peak - base

    shape = profiles.category_shape(category, index, site).to_numpy()
    sensitivity = profiles.seasonal_sensitivity(category)
    seasonal = 1.0 +  sensitivity * (envelope.to_numpy() - 1.0)
    activity = _daily_activity(index, rng)

    power = base + span * NOMINAL_UTILISATION * shape * seasonal * activity
    power = power * rng.normal(1.0, MEASUREMENT_NOISE, size = len(index))

    power = np.clip(power, base * MIN_UNDERSHOOT, peak * MAX_OVERSHOOT)
    return pd.Series(power, index=index, name="power_kw")

