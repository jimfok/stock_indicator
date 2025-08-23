"""Utility functions for calculating stock market technical indicators.

Adjusted closing prices are recommended for all calculations to account for
corporate actions such as dividends and stock splits.
"""
# TODO: review

from __future__ import annotations

from typing import List

import pandas


def sma(price_series: pandas.Series, window_size: int) -> pandas.Series:
    """Calculate the Simple Moving Average (SMA).

    Adjusted close prices are the recommended input.

    Parameters
    ----------
    price_series: pandas.Series
        Series of prices, preferably adjusted close values.
    window_size: int
        Number of periods to include in the moving average.

    Returns
    -------
    pandas.Series
        Simple moving average of the provided price series.
    """
    return price_series.rolling(window=window_size).mean()


def ema(price_series: pandas.Series, window_size: int) -> pandas.Series:
    """Calculate the Exponential Moving Average (EMA).

    Adjusted close prices are the recommended input.

    Parameters
    ----------
    price_series: pandas.Series
        Series of prices, preferably adjusted close values.
    window_size: int
        Number of periods for exponential weighting.

    Returns
    -------
    pandas.Series
        Exponential moving average of the provided price series.
    """
    return price_series.ewm(span=window_size, adjust=False).mean()


# TODO: review
def relative_strength(
    price_series: pandas.Series, benchmark_series: pandas.Series
) -> pandas.Series:
    """Calculate relative strength versus a benchmark index.

    Parameters
    ----------
    price_series: pandas.Series
        Series of asset prices.
    benchmark_series: pandas.Series
        Series of benchmark index prices.

    Returns
    -------
    pandas.Series
        Ratio of the asset price to the benchmark price with aligned indexes.
    """
    aligned_price_series, aligned_benchmark_series = price_series.align(
        benchmark_series, join="inner"
    )
    return aligned_price_series.divide(aligned_benchmark_series)


def macd(
    price_series: pandas.Series,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> pandas.DataFrame:
    """Calculate the Moving Average Convergence Divergence (MACD).

    Adjusted close prices are the recommended input.

    Parameters
    ----------
    price_series: pandas.Series
        Series of prices, preferably adjusted close values.
    fast_window: int
        Window length for the fast EMA.
    slow_window: int
        Window length for the slow EMA.
    signal_window: int
        Window length for the signal line.

    Returns
    -------
    pandas.DataFrame
        Data frame containing the MACD line, signal line, and histogram.
    """
    fast_ema_series = price_series.ewm(span=fast_window, adjust=False).mean()
    slow_ema_series = price_series.ewm(span=slow_window, adjust=False).mean()
    macd_line_series = fast_ema_series - slow_ema_series
    signal_line_series = macd_line_series.ewm(span=signal_window, adjust=False).mean()
    histogram_series = macd_line_series - signal_line_series
    return pandas.DataFrame(
        {
            "macd": macd_line_series,
            "signal": signal_line_series,
            "histogram": histogram_series,
        }
    )


def rsi(price_series: pandas.Series, window_size: int = 14) -> pandas.Series:
    """Calculate the Relative Strength Index (RSI).

    Adjusted close prices are the recommended input.

    Parameters
    ----------
    price_series: pandas.Series
        Series of prices, preferably adjusted close values.
    window_size: int, default 14
        Number of periods to use for the calculation.

    Returns
    -------
    pandas.Series
        Relative Strength Index values.
    """
    price_change_series = price_series.diff()
    gain_series = price_change_series.clip(lower=0)
    loss_series = -price_change_series.clip(upper=0)
    average_gain_series = gain_series.ewm(alpha=1 / window_size, adjust=False).mean()
    average_loss_series = loss_series.ewm(alpha=1 / window_size, adjust=False).mean()
    relative_strength_series = average_gain_series / average_loss_series
    return 100 - (100 / (1 + relative_strength_series))


def cci(
    high_price_series: pandas.Series,
    low_price_series: pandas.Series,
    close_price_series: pandas.Series,
    window_size: int = 20,
) -> pandas.Series:
    """Calculate the Commodity Channel Index (CCI).

    Adjusted price data, especially adjusted close, is recommended.

    Parameters
    ----------
    high_price_series: pandas.Series
        Series of high prices.
    low_price_series: pandas.Series
        Series of low prices.
    close_price_series: pandas.Series
        Series of closing prices, preferably adjusted close values.
    window_size: int, default 20
        Number of periods to use for the calculation.

    Returns
    -------
    pandas.Series
        Commodity Channel Index values.
    """
    typical_price_series = (
        high_price_series + low_price_series + close_price_series
    ) / 3
    moving_average_series = typical_price_series.rolling(window=window_size).mean()
    mean_deviation_series = typical_price_series.rolling(window=window_size).apply(
        lambda price_window: (abs(price_window - price_window.mean())).mean(),
        raw=False,
    )
    return (typical_price_series - moving_average_series) / (
        0.015 * mean_deviation_series
    )
# TODO: review

def kalman_filter(
    price_series: pandas.Series,
    process_variance: float = 1e-5,
    observation_variance: float = 1.0,
) -> pandas.DataFrame:
    """Apply a simple Kalman filter to a price series.

    The function returns the smoothed estimate and one standard deviation
    bounds around that estimate.

    Parameters
    ----------
    price_series: pandas.Series
        Series of observed prices, typically adjusted closing values.
    process_variance: float, default 1e-5
        Expected variance in the underlying process.
    observation_variance: float, default 1.0
        Expected variance in the observation noise.

    Returns
    -------
    pandas.DataFrame
        Data frame containing the filtered estimate (``estimate``) and
        upper and lower bounds (``upper_bound`` and ``lower_bound``).
    """
    estimated_price: float = 0.0
    estimation_error: float = 1.0
    estimate_list: List[float] = []
    upper_bound_list: List[float] = []
    lower_bound_list: List[float] = []
    for observed_price in price_series:
        predicted_estimate = estimated_price
        predicted_error = estimation_error + process_variance
        kalman_gain = predicted_error / (predicted_error + observation_variance)
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
        index=price_series.index,
    )
