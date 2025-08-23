"""Tests for technical indicator functions."""
# TODO: review

import os
import sys

import numpy
import pandas
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator.indicators import (
    cci,
    ema,
    kalman_filter,
    macd,
    relative_strength,
    rsi,
    sma,
)


def test_sma_calculates_simple_average() -> None:
    price_series = pandas.Series([1, 2, 3, 4, 5])
    result_series = sma(price_series, window_size=3)
    expected_series = pandas.Series([numpy.nan, numpy.nan, 2.0, 3.0, 4.0])
    pandas.testing.assert_series_equal(result_series, expected_series)


def test_ema_matches_pandas_calculation() -> None:
    price_series = pandas.Series([1, 2, 3, 4, 5])
    result_series = ema(price_series, window_size=3)
    expected_series = price_series.ewm(span=3, adjust=False).mean()
    pandas.testing.assert_series_equal(result_series, expected_series)


def test_macd_returns_expected_components() -> None:
    price_series = pandas.Series(range(1, 11))
    result_dataframe = macd(price_series)
    expected_macd_series = ema(price_series, window_size=12) - ema(price_series, window_size=26)
    expected_signal_series = expected_macd_series.ewm(span=9, adjust=False).mean()
    expected_histogram_series = expected_macd_series - expected_signal_series
    expected_macd_series.name = "macd"
    expected_signal_series.name = "signal"
    expected_histogram_series.name = "histogram"
    pandas.testing.assert_series_equal(result_dataframe["macd"], expected_macd_series)
    pandas.testing.assert_series_equal(result_dataframe["signal"], expected_signal_series)
    pandas.testing.assert_series_equal(
        result_dataframe["histogram"], expected_histogram_series
    )


def test_rsi_matches_reference_formula() -> None:
    price_series = pandas.Series([1, 2, 3, 2, 2, 3, 4, 5, 4, 4])
    result_series = rsi(price_series, window_size=14)
    price_change_series = price_series.diff()
    gain_series = price_change_series.clip(lower=0)
    loss_series = -price_change_series.clip(upper=0)
    average_gain_series = gain_series.ewm(alpha=1 / 14, adjust=False).mean()
    average_loss_series = loss_series.ewm(alpha=1 / 14, adjust=False).mean()
    relative_strength_series = average_gain_series / average_loss_series
    expected_series = 100 - (100 / (1 + relative_strength_series))
    pandas.testing.assert_series_equal(result_series, expected_series)


# TODO: review
def test_relative_strength_returns_ratio() -> None:
    symbol_price_series = pandas.Series(
        [10, 20, 30], index=pandas.date_range("2020-01-01", periods=3, freq="D")
    )
    benchmark_price_series = pandas.Series(
        [5, 10, 15], index=pandas.date_range("2020-01-02", periods=3, freq="D")
    )
    result_series = relative_strength(symbol_price_series, benchmark_price_series)
    aligned_symbol_series, aligned_benchmark_series = symbol_price_series.align(
        benchmark_price_series, join="inner"
    )
    expected_series = aligned_symbol_series.divide(aligned_benchmark_series)
    pandas.testing.assert_series_equal(result_series, expected_series)


def test_cci_matches_manual_calculation() -> None:
    high_price_series = pandas.Series([1, 2, 3, 4, 5])
    low_price_series = pandas.Series([1, 1, 2, 3, 4])
    close_price_series = pandas.Series([1, 2, 3, 4, 5])
    result_series = cci(
        high_price_series,
        low_price_series,
        close_price_series,
        window_size=3,
    )
    typical_price_series = (
        high_price_series + low_price_series + close_price_series
    ) / 3
    moving_average_series = typical_price_series.rolling(window=3).mean()
    mean_deviation_series = typical_price_series.rolling(window=3).apply(
        lambda price_window: (abs(price_window - price_window.mean())).mean(),
        raw=False,
    )
    expected_series = (typical_price_series - moving_average_series) / (
        0.015 * mean_deviation_series
    )
    pandas.testing.assert_series_equal(result_series, expected_series)


def test_kalman_filter_produces_bounds() -> None:
    price_series = pandas.Series([1.0, 2.0, 3.0])

    def manual_kalman(price_values: pandas.Series) -> pandas.DataFrame:
        estimated_price: float = 0.0
        estimation_error: float = 1.0
        estimate_list: List[float] = []
        upper_bound_list: List[float] = []
        lower_bound_list: List[float] = []
        for observed_price in price_values:
            predicted_estimate = estimated_price
            predicted_error = estimation_error + 1e-5
            kalman_gain = predicted_error / (predicted_error + 1.0)
            estimated_price = predicted_estimate + kalman_gain * (
                observed_price - predicted_estimate
            )
            estimation_error = (1 - kalman_gain) * predicted_error
            standard_deviation = estimation_error ** 0.5
            estimate_list.append(estimated_price)
            upper_bound_list.append(estimated_price + standard_deviation)
            lower_bound_list.append(estimated_price - standard_deviation)
        return pandas.DataFrame(
            {
                "estimate": estimate_list,
                "upper_bound": upper_bound_list,
                "lower_bound": lower_bound_list,
            },
            index=price_values.index,
        )

    result_dataframe = kalman_filter(price_series)
    expected_dataframe = manual_kalman(price_series)
    pandas.testing.assert_frame_equal(result_dataframe, expected_dataframe)
