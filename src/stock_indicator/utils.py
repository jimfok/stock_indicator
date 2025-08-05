"""Utility helpers for the stock_indicator package."""

from typing import Sequence


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
