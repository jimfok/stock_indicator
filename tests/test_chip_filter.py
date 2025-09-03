"""Tests for chip concentration and volume profile features."""
# TODO: review

from __future__ import annotations

import os
import sys
import numpy
import pandas

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator.chip_filter import (
    calculate_chip_concentration_metrics,
    calculate_volume_profile_features,
)
from stock_indicator.volume_profile import VolumeProfile


def test_calculate_volume_profile_features_returns_expected_metrics() -> None:
    """Volume profile metrics should match expected values."""

    probabilities = numpy.array([0.08, 0.4, 0.02, 0.4, 0.08, 0.02])
    volume_profile = VolumeProfile(
        point_of_control=1.5,
        value_area_high=4.5,
        value_area_low=0.5,
        bin_edges=numpy.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        probability_distribution=probabilities,
    )
    result = calculate_volume_profile_features(volume_profile, current_price=2.5)

    assert numpy.isclose(result["hhi"], 0.3336, atol=1e-4)
    assert numpy.isclose(result["distance_to_poc"], 1.0)
    assert numpy.isclose(result["above_volume_ratio_vp"], 0.5)
    assert numpy.isclose(result["below_volume_ratio_vp"], 0.48)
    assert result["hvn_count"] == 2.0
    assert numpy.isclose(result["lvn_depth"], 0.015, atol=1e-3)


def test_calculate_chip_concentration_metrics_includes_volume_profile_metrics() -> None:
    """The concentration metric function should include volume profile features."""

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    close_values = [float(value) for value in range(60, 120)]
    high_values = close_values
    low_values = [value - 1.0 for value in close_values]
    volume_values = [100 for _ in range(60)]
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

    metrics = calculate_chip_concentration_metrics(
        ohlcv, lookback_window_size=60, include_volume_profile=True
    )

    expected_keys = {
        "price_score",
        "near_price_volume_ratio",
        "above_price_volume_ratio",
        "histogram_node_count",
        "hhi",
        "distance_to_poc",
        "above_volume_ratio_vp",
        "below_volume_ratio_vp",
        "hvn_count",
        "lvn_depth",
    }
    assert expected_keys.issubset(metrics.keys())
    for key in expected_keys:
        assert metrics[key] is not None


def test_calculate_chip_concentration_metrics_excludes_volume_profile_by_default() -> None:
    """Volume profile metrics are ``None`` unless explicitly requested."""

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    close_values = [float(value) for value in range(60, 120)]
    high_values = close_values
    low_values = [value - 1.0 for value in close_values]
    volume_values = [100 for _ in range(60)]
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

    metrics = calculate_chip_concentration_metrics(ohlcv, lookback_window_size=60)

    assert metrics["hhi"] is None
    assert metrics["distance_to_poc"] is None
    assert metrics["above_volume_ratio_vp"] is None
    assert metrics["below_volume_ratio_vp"] is None
    assert metrics["hvn_count"] is None
    assert metrics["lvn_depth"] is None
