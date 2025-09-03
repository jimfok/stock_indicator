from __future__ import annotations

import numpy
import pandas

from .volume_profile import VolumeProfile, calculate_volume_profile


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


# TODO: review
def calculate_volume_profile_features(
    volume_profile: VolumeProfile, current_price: float
) -> dict[str, float]:
    """Derive metrics from a volume profile for chip concentration analysis.

    Parameters
    ----------
    volume_profile:
        ``VolumeProfile`` instance containing the histogram distribution.
    current_price:
        Latest traded price.

    Returns
    -------
    dict[str, float]
        Dictionary of ``hhi``, ``distance_to_poc``, ``above_volume_ratio_vp``,
        ``below_volume_ratio_vp``, ``hvn_count`` and ``lvn_depth`` values.
    """
    probabilities = volume_profile.probability_distribution
    bin_edges = volume_profile.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    herfindahl_hirschman_index = float(numpy.sum(probabilities ** 2))

    bin_widths = numpy.diff(bin_edges)
    average_true_range = float(bin_widths[0]) if bin_widths.size > 0 else 1.0
    if average_true_range <= 0:
        normalized_distance = 0.0
    else:
        normalized_distance = float(
            (current_price - volume_profile.point_of_control) / average_true_range
        )

    above_volume_ratio_vp = float(probabilities[bin_centers > current_price].sum())
    below_volume_ratio_vp = float(probabilities[bin_centers < current_price].sum())

    smoothing_kernel = numpy.array([0.25, 0.5, 0.25])
    smoothed_probabilities = numpy.convolve(probabilities, smoothing_kernel, mode="same")

    peak_indices: list[int] = []
    valley_indices: list[int] = [0]
    for index in range(1, smoothed_probabilities.size - 1):
        left_value = smoothed_probabilities[index - 1]
        middle_value = smoothed_probabilities[index]
        right_value = smoothed_probabilities[index + 1]
        if middle_value > left_value and middle_value > right_value:
            peak_indices.append(index)
        if middle_value < left_value and middle_value < right_value:
            valley_indices.append(index)
    valley_indices.append(smoothed_probabilities.size - 1)

    prominence_threshold = 0.01 * float(smoothed_probabilities.max())
    high_volume_node_count = 0
    for peak_index in peak_indices:
        left_valley = max([v for v in valley_indices if v < peak_index], default=None)
        right_valley = min([v for v in valley_indices if v > peak_index], default=None)
        if left_valley is None or right_valley is None:
            continue
        prominence = min(
            smoothed_probabilities[peak_index] - smoothed_probabilities[left_valley],
            smoothed_probabilities[peak_index] - smoothed_probabilities[right_valley],
        )
        if prominence >= prominence_threshold:
            high_volume_node_count += 1

    valley_depths: list[float] = []
    for valley_index in valley_indices[1:-1]:
        left_peak = max([p for p in peak_indices if p < valley_index], default=None)
        right_peak = min([p for p in peak_indices if p > valley_index], default=None)
        if left_peak is None or right_peak is None:
            continue
        depth = min(
            smoothed_probabilities[left_peak],
            smoothed_probabilities[right_peak],
        ) - smoothed_probabilities[valley_index]
        valley_depths.append(float(depth))
    lowest_volume_node_depth = max(valley_depths) if valley_depths else 0.0

    return {
        "hhi": herfindahl_hirschman_index,
        "distance_to_poc": normalized_distance,
        "above_volume_ratio_vp": above_volume_ratio_vp,
        "below_volume_ratio_vp": below_volume_ratio_vp,
        "hvn_count": float(high_volume_node_count),
        "lvn_depth": float(lowest_volume_node_depth),
    }


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
    try:
        volume_profile = calculate_volume_profile(
            window_frame, lookback_window_size=lookback_window_size
        )
        volume_profile_metrics = calculate_volume_profile_features(
            volume_profile, current_price
        )
    except ValueError:
        volume_profile_metrics = {
            "hhi": None,
            "distance_to_poc": None,
            "above_volume_ratio_vp": None,
            "below_volume_ratio_vp": None,
            "hvn_count": None,
            "lvn_depth": None,
        }

    return {
        "price_score": price_score,
        "near_price_volume_ratio": near_volume_ratio,
        "above_price_volume_ratio": above_volume_ratio,
        "histogram_node_count": histogram_node_count,
        **volume_profile_metrics,
    }
