"""Utilities for computing a volume profile over a price series."""
# TODO: review

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy
import pandas


@dataclass
class VolumeProfile:
    """Representation of a volume profile for a series of prices."""

    point_of_control: float
    value_area_high: float
    value_area_low: float
    bin_edges: numpy.ndarray
    probability_distribution: numpy.ndarray


def calculate_volume_profile(
    ohlcv: pandas.DataFrame, lookback_window_size: int
) -> VolumeProfile:
    """Calculate a volume profile with adaptive bin sizing.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        Data frame containing ``high``, ``low``, ``close``, and ``volume`` columns.
    lookback_window_size : int
        Window size used to compute the average true range for determining bin width.

    Returns
    -------
    VolumeProfile
        A dataclass containing the point of control, value area high, value area low,
        bin edges, and the probability distribution of volume across price levels.

    Raises
    ------
    ValueError
        If required columns are missing, the data frame is empty, or the lookback
        window size is not positive.
    """
    required_columns = {"high", "low", "close", "volume"}
    if not required_columns.issubset(ohlcv.columns):
        missing = required_columns - set(ohlcv.columns)
        raise ValueError(
            "Data frame must contain columns: " + ", ".join(sorted(missing))
        )
    if ohlcv.empty:
        raise ValueError("Input data frame must not be empty")
    if lookback_window_size <= 0:
        raise ValueError("lookback_window_size must be positive")

    high_series = ohlcv["high"]
    low_series = ohlcv["low"]
    close_series = ohlcv["close"]
    previous_close_series = close_series.shift(1)

    true_range = pandas.concat(
        [
            high_series - low_series,
            (high_series - previous_close_series).abs(),
            (low_series - previous_close_series).abs(),
        ],
        axis=1,
    ).max(axis=1)

    average_true_range = (
        true_range.rolling(window=lookback_window_size).mean().iloc[-1]
    )
    highest_price = float(high_series.max())
    lowest_price = float(low_series.min())
    price_range = highest_price - lowest_price

    if pandas.isna(average_true_range) or average_true_range <= 0:
        bin_size = price_range * 0.05
    else:
        bin_size = float(average_true_range)
    if bin_size <= 0:
        bin_size = price_range if price_range > 0 else 1.0

    number_of_bins = max(int(math.ceil(price_range / bin_size)), 1)

    typical_price = (high_series + low_series) / 2
    volume_values = ohlcv["volume"].to_numpy()
    histogram, bin_edges = numpy.histogram(
        typical_price,
        bins=number_of_bins,
        range=(lowest_price, highest_price),
        weights=volume_values,
    )
    total_volume = histogram.sum()
    if total_volume > 0:
        probability_distribution = histogram / total_volume
    else:
        probability_distribution = numpy.zeros_like(histogram, dtype=float)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    point_of_control_index = int(numpy.argmax(probability_distribution))
    point_of_control_price = float(bin_centers[point_of_control_index])

    target_probability = 0.7
    cumulative_probability = probability_distribution[point_of_control_index]
    lower_index = point_of_control_index
    upper_index = point_of_control_index
    left_index = point_of_control_index - 1
    right_index = point_of_control_index + 1
    while (
        cumulative_probability < target_probability
        and (left_index >= 0 or right_index < probability_distribution.size)
    ):
        left_probability = (
            probability_distribution[left_index] if left_index >= 0 else -1.0
        )
        right_probability = (
            probability_distribution[right_index]
            if right_index < probability_distribution.size
            else -1.0
        )
        if right_probability >= left_probability:
            cumulative_probability += right_probability
            upper_index = right_index
            right_index += 1
        else:
            cumulative_probability += left_probability
            lower_index = left_index
            left_index -= 1

    value_area_low_price = float(bin_edges[lower_index])
    value_area_high_price = float(bin_edges[upper_index + 1])

    return VolumeProfile(
        point_of_control=point_of_control_price,
        value_area_high=value_area_high_price,
        value_area_low=value_area_low_price,
        bin_edges=bin_edges,
        probability_distribution=probability_distribution,
    )
