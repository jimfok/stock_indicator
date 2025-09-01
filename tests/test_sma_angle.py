import math
import pandas
import pytest

import stock_indicator.strategy as strategy_module


def test_sma_angle_computation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A known percentage change should produce the expected angle."""
    # TODO: review

    price_data_frame = pandas.DataFrame({"open": [1.0, 1.0], "close": [1.0, 1.0]})

    def fake_attach(
        data_frame: pandas.DataFrame,
        window_size: int = 40,
        require_close_above_long_term_sma: bool = True,
        sma_window_factor: float | None = None,
    ) -> None:
        data_frame["sma_value"] = pandas.Series([100.0, 110.0])
        data_frame["sma_previous"] = data_frame["sma_value"].shift(1)
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series([False, True])
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series([False, False])

    monkeypatch.setattr(strategy_module, "attach_ema_sma_cross_signals", fake_attach)

    strategy_module.attach_ema_sma_cross_with_slope_signals(
        price_data_frame, angle_range=(-90.0, 90.0)
    )

    expected_angle = math.degrees(math.atan(0.1))
    assert price_data_frame.loc[1, "sma_angle"] == pytest.approx(expected_angle)
