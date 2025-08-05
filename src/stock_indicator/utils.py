"""Utility helpers for the stock_indicator package."""

import datetime
from typing import Sequence

import pandas as pd
import yfinance as yf


def load_stock_history(symbol: str, interval: str, decimals: int = 3) -> pd.DataFrame:
    """Download historical stock data and normalize column values.

    Args:
        symbol: Ticker symbol to fetch.
        interval: Data interval accepted by :func:`yfinance.download`.
        decimals: Number of decimal places to round OHLC values.

    Returns:
        DataFrame with the stock history. The index is reset to expose a
        ``Date`` column and OHLC values are rounded to the specified
        precision. If no data is returned, the empty DataFrame is
        returned unchanged.
    """

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365 * 100)
    df_stock = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    if df_stock.empty:
        return df_stock

    df_stock = df_stock.round({"Low": decimals, "High": decimals, "Close": decimals, "Open": decimals})
    df_stock.reset_index(inplace=True)
    return df_stock


def validate_series(series: Sequence[float], min_length: int = 1) -> None:
    """Validate that the provided series contains numeric values and meets the minimum length.

    Args:
        series: Sequence of numeric values.
        min_length: Minimum required length for the series.

    Raises:
        NotImplementedError: Placeholder for validation logic.
    """
    raise NotImplementedError("Series validation not implemented.")


def to_decimal(value):
    """Convert value to Decimal for high-precision calculations.

    Args:
        value: The input value to convert.

    Returns:
        The converted Decimal value.
    """
    raise NotImplementedError("Decimal conversion not implemented.")
