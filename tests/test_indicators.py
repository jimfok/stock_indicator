"""Tests for technical indicator functions."""
# TODO: review

import os
import sys

import numpy
import pandas

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator.indicators import cci, ema, macd, rsi, sma


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
    pandas.testing.assert_series_equal(result_dataframe["macd"], expected_macd_series)
    pandas.testing.assert_series_equal(result_dataframe["signal"], expected_signal_series)
    pandas.testing.assert_series_equal(result_dataframe["histogram"], expected_histogram_series)


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
