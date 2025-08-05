"""Technical indicator calculation stubs."""

from typing import Sequence

def ema(prices: Sequence[float], period: int = 12):
    """Calculate the Exponential Moving Average (EMA).

    Args:
        prices: Sequence of price values.
        period: Number of periods to use for the EMA calculation.

    Returns:
        The EMA values.
    """
    raise NotImplementedError("EMA calculation not implemented.")


def rsi(prices: Sequence[float], period: int = 14):
    """Calculate the Relative Strength Index (RSI).

    Args:
        prices: Sequence of price values.
        period: Number of periods to use for the RSI calculation.

    Returns:
        The RSI values.
    """
    raise NotImplementedError("RSI calculation not implemented.")


def sma(prices: Sequence[float], period: int = 14):
    """Calculate the Simple Moving Average (SMA).

    Args:
        prices: Sequence of price values.
        period: Number of periods to use for the SMA calculation.

    Returns:
        The SMA values.
    """
    raise NotImplementedError("SMA calculation not implemented.")
