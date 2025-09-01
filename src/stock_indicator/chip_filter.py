from __future__ import annotations

import numpy
import pandas


# TODO: review
NEAR_VOLUME_RATIO_MAX: float = 0.12
# TODO: review
ABOVE_VOLUME_RATIO_MAX: float = 0.10


# TODO: review
def passes_chip_concentration_filter(
    near_volume_ratio: float | None, above_volume_ratio: float | None
) -> bool:
    """Validate chip concentration metrics against loose thresholds.

    Parameters
    ----------
    near_volume_ratio:
        Fraction of volume within the near-price band.
    above_volume_ratio:
        Fraction of volume above the current price.

    Returns
    -------
    bool
        ``True`` when both ratios are within ``[0, 1]`` and do not exceed the
        ``loose`` thresholds. Otherwise ``False``.
    """
    if near_volume_ratio is None or above_volume_ratio is None:
        return False
    if not (0.0 <= near_volume_ratio <= 1.0):
        return False
    if not (0.0 <= above_volume_ratio <= 1.0):
        return False
    return (
        near_volume_ratio <= NEAR_VOLUME_RATIO_MAX
        and above_volume_ratio <= ABOVE_VOLUME_RATIO_MAX
    )


def calculate_chip_concentration_metrics(
    ohlcv: pandas.DataFrame,
    lookback_window_size: int = 60,
    bin_count: int = 50,
    near_price_band_ratio: float = 0.03,
) -> dict[str, float | int | None]:
    """Calculate chip concentration metrics for a price series.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        Data frame with ``high``, ``low``, ``close`` and ``volume`` columns.
    lookback_window_size : int, default 60
        Number of rows to include in the calculation window.
    bin_count : int, default 50
        Number of price bins used for the histogram.
    near_price_band_ratio : float, default 0.03
        Fractional width around the current price for the near-price band.

    Returns
    -------
    dict[str, float | int | None]
        Dictionary containing ``price_score``, ``near_price_volume_ratio``,
        ``above_price_volume_ratio`` and ``histogram_node_count``. Values are
        ``None`` when the data is insufficient.
    """
    required_columns = {"high", "low", "close", "volume"}
    if not required_columns.issubset(ohlcv.columns):
        return {
            "price_score": None,
            "near_price_volume_ratio": None,
            "above_price_volume_ratio": None,
            "histogram_node_count": None,
        }
    window_frame = ohlcv.tail(lookback_window_size).copy()
    if (
        len(window_frame) < max(lookback_window_size, 30)
        or float(window_frame["volume"].sum()) <= 0.0
    ):
        return {
            "price_score": None,
            "near_price_volume_ratio": None,
            "above_price_volume_ratio": None,
            "histogram_node_count": None,
        }
    typical_price_series = (
        window_frame["high"] + window_frame["low"] + window_frame["close"]
    ) / 3.0
    volume_array = window_frame["volume"].astype(float).to_numpy()
    lowest_price = float(window_frame["low"].min())
    highest_price = float(window_frame["high"].max())
    bin_edges = numpy.linspace(lowest_price, highest_price, bin_count + 1)
    bin_indices = numpy.clip(
        numpy.digitize(typical_price_series, bin_edges) - 1, 0, bin_count - 1
    )
    histogram = numpy.zeros(bin_count)
    numpy.add.at(histogram, bin_indices, volume_array)
    total_volume = float(histogram.sum())
    if total_volume <= 0.0:
        return {
            "price_score": None,
            "near_price_volume_ratio": None,
            "above_price_volume_ratio": None,
            "histogram_node_count": None,
        }
    probabilities = histogram / total_volume
    herfindahl_index = float(numpy.sum(probabilities ** 2))
    price_score = float(
        (herfindahl_index - 1.0 / bin_count) / (1.0 - 1.0 / bin_count)
    )
    current_price = float(window_frame["close"].iloc[-1])
    near_mask = typical_price_series.between(
        current_price * (1 - near_price_band_ratio),
        current_price * (1 + near_price_band_ratio),
    ).to_numpy()
    near_volume_ratio = float(volume_array[near_mask].sum() / total_volume)
    above_volume_ratio = float(
        volume_array[(typical_price_series > current_price).to_numpy()].sum()
        / total_volume
    )
    peak_mask = (histogram > numpy.median(histogram)) & (histogram > 0)
    peak_boundaries = numpy.diff(
        numpy.concatenate(([False], peak_mask, [False]))
    ) == 1
    histogram_node_count = int(numpy.sum(peak_boundaries))
    return {
        "price_score": price_score,
        "near_price_volume_ratio": near_volume_ratio,
        "above_price_volume_ratio": above_volume_ratio,
        "histogram_node_count": histogram_node_count,
    }
