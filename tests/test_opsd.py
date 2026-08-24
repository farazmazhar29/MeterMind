"""Tests for the time-shift arithmetic.

This is pinned down because it is the one piece of Week 1 logic that is quietly
wrong-able: an off-by-one in the offset silently misaligns every weekday in the
dataset, and nothing downstream would raise an error -- the synthetic sites would
just have their weekends on the wrong days.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from metermind.data import opsd


def utc(year, month, day, hour=0):
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


class TestComputeTimeOffset:
    def test_offset_is_whole_weeks(self):
        """The whole point: weekday alignment must survive the shift."""
        offset = opsd.compute_time_offset(utc(2020, 9, 30), utc(2026, 8, 14))
        assert offset.days % 7 == 0

    def test_offset_is_multiple_of_364_days(self):
        offset = opsd.compute_time_offset(utc(2020, 9, 30), utc(2026, 8, 14))
        assert offset.days % 364 == 0

    def test_weekday_is_preserved(self):
        source_end = utc(2020, 9, 30)
        offset = opsd.compute_time_offset(source_end, utc(2026, 8, 14))
        assert (source_end + offset).weekday() == source_end.weekday()

    def test_rounds_up_past_the_target(self):
        """Must cover the target, not stop short of it.

        Rounding down would leave the shifted series ending ~11 months before
        today, so every "last week" question would have no data.
        """
        target = utc(2026, 8, 14)
        offset = opsd.compute_time_offset(utc(2020, 9, 30), target)
        assert utc(2020, 9, 30) + offset >= target

    def test_seasonal_drift_stays_under_two_weeks(self):
        """364-day steps drift ~1.25 days a year against the calendar."""
        source_end = utc(2020, 9, 30)
        shifted = source_end + opsd.compute_time_offset(source_end, utc(2026, 8, 14))
        day_of_year_drift = abs(shifted.timetuple().tm_yday - source_end.timetuple().tm_yday)
        assert day_of_year_drift < 14

    def test_no_shift_when_data_already_recent(self):
        assert opsd.compute_time_offset(utc(2026, 12, 1), utc(2026, 8, 14)) == timedelta(0)

    def test_naive_datetimes_are_treated_as_utc(self):
        aware = opsd.compute_time_offset(utc(2020, 9, 30), utc(2026, 8, 14))
        naive = opsd.compute_time_offset(datetime(2020, 9, 30), datetime(2026, 8, 14))
        assert aware == naive


class TestSeasonalEnvelope:
    @pytest.fixture
    def fake_load(self):
        """Two years of hourly load: seasonal swing plus a strong daily cycle."""
        index = pd.date_range("2019-01-01", "2020-12-31", freq="h", tz="UTC")
        seasonal = 50_000 + 10_000 * np.cos(2 * np.pi * index.dayofyear.to_numpy() / 365)
        daily = 8_000 * np.sin(2 * np.pi * (index.hour.to_numpy() - 6) / 24)
        return pd.Series(seasonal + daily, index=index, name="load_mw")

    def test_centred_on_one(self, fake_load):
        index = pd.date_range("2019-06-01", "2019-06-30", freq="15min", tz="UTC")
        envelope = opsd.seasonal_envelope(fake_load, index, timedelta(0))
        assert 0.7 < envelope.mean() < 1.3

    def test_matches_target_index_with_no_gaps(self, fake_load):
        index = pd.date_range("2019-06-01", "2019-06-30", freq="15min", tz="UTC")
        envelope = opsd.seasonal_envelope(fake_load, index, timedelta(0))
        assert envelope.index.equals(index)
        assert not envelope.isna().any()

    def test_intra_day_cycle_is_stripped(self, fake_load):
        """The critical property.

        The source has an 8,000 MW daily swing on a 50,000 MW base. If any of it
        survives into the envelope, every synthetic site inherits the national
        evening peak and all eight end up near-perfectly correlated.
        """
        index = pd.date_range("2019-06-01", "2019-06-30", freq="15min", tz="UTC")
        envelope = opsd.seasonal_envelope(fake_load, index, timedelta(0))

        by_hour = envelope.groupby(envelope.index.hour).mean()
        assert (by_hour.max() - by_hour.min()) < 0.02

    def test_offset_shifts_the_signal(self, fake_load):
        index = pd.date_range("2020-06-01", "2020-06-15", freq="15min", tz="UTC")
        unshifted = opsd.seasonal_envelope(fake_load, index, timedelta(0))
        shifted = opsd.seasonal_envelope(fake_load, index, timedelta(days=364))
        assert not unshifted.equals(shifted)
