"""Tests for strategy choice splitting utility."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import stock_indicator.strategy as strategy


def test_split_strategy_choices_preserves_internal_commas() -> None:
    """Ensure commas used for numeric parameters are not treated as separators."""
    complex_name = "ema_sma_cross_testing_4_-0.01_65_0.05,0.0802_0.83,1.00"
    tokens = strategy._split_strategy_choices(complex_name)
    assert tokens == [complex_name]


def test_split_strategy_choices_splits_on_or_token() -> None:
    """Verify that recognized separators split strategy expressions."""
    composite_name = "first or second"
    tokens = strategy._split_strategy_choices(composite_name)
    assert tokens == ["first", "second"]
