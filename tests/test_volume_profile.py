"""Tests for the volume profile calculation."""
# TODO: review

from __future__ import annotations

import numpy
import pandas

from stock_indicator.volume_profile import calculate_volume_profile, VolumeProfile


def test_calculate_volume_profile_returns_expected_profile() -> None:
    """The function should return a volume profile with correct key values."""

    date_index = pandas.date_range("2020-01-01", periods=5, freq="D")
    close_values = [10.0, 11.0, 12.0, 13.0, 14.0]
    high_values = close_values
    low_values = [value - 1.0 for value in close_values]
    volume_values = [100, 200, 300, 200, 100]

    ohlcv = pandas.DataFrame(
        {
            "Date": date_index,
            "open": close_values,
            "high": high_values,
            "low": low_values,
            "close": close_values,
            "volume": volume_values,
        }
    )

    result = calculate_volume_profile(ohlcv, lookback_window_size=5)

    assert isinstance(result, VolumeProfile)
    assert numpy.isclose(result.point_of_control, 11.5)
    assert numpy.isclose(result.value_area_low, 10.0)
    assert numpy.isclose(result.value_area_high, 13.0)
    assert numpy.allclose(result.probability_distribution.sum(), 1.0)
    assert len(result.bin_edges) == len(result.probability_distribution) + 1
