"""Scheduled daily tasks for updating data and evaluating strategies."""
# TODO: review

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas

from .symbols import update_symbol_cache, load_symbols, load_yf_symbols
from .data_loader import download_history
from .strategy import SUPPORTED_STRATEGIES, load_ff12_groups_by_symbol

LOGGER = logging.getLogger(__name__)


def parse_daily_task_arguments(argument_line: str) -> Tuple[
    float | None,
    int | None,
    str,
    str,
    float,
    set[int] | None,
]:
    """Parse a cron job argument string.

    The expected format is ``dollar_volume>NUMBER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``,
    ``dollar_volume>NUMBER% BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``,
    ``dollar_volume=TopN BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]`` (or legacy ``Nth``),
    ``dollar_volume>NUMBER,TopN BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]`` (or 
    legacy ``,Nth``), or ``dollar_volume>NUMBER%,TopN BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``
    (or legacy ``,Nth``). Matching is case-insensitive for ``TopN``.

    Parameters
    ----------
    argument_line: str
        Argument string describing filters and strategy names.

    Returns
    -------
    Tuple[float | None, int | None, str, str, float]
        Tuple containing either a minimum dollar volume threshold in millions
        or a ratio of the total market, followed by the ranking position, the
        buy strategy name, the sell strategy name, and the stop loss
        percentage.
    """
    argument_parts = argument_line.split()
    if len(argument_parts) not in (3, 4, 5):
        raise ValueError(
            "argument_line must be 'dollar_volume>NUMBER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', "
            "'dollar_volume>NUMBER% BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', "
            "'dollar_volume=TopN BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]' (or 'Nth'), "
            "'dollar_volume>NUMBER,TopN' (or ',Nth'), or 'dollar_volume>NUMBER%,TopN' (or ',Nth')",
        )
    # Allow an optional group=1,2,... token to appear anywhere in the first
    # few arguments. Extract it if present, then parse the remaining tokens as
    # volume filter, buy strategy, sell strategy, [stop loss].
    allowed_groups: set[int] | None = None
    tokens: list[str] = []
    for token in argument_parts:
        if token.startswith("group="):
            try:
                raw = token.split("=", 1)[1]
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                parsed = {int(p) for p in parts}
            except ValueError as parse_error:  # noqa: BLE001
                raise ValueError("Invalid group list; expected integers 1-11") from parse_error
            if any(identifier < 1 or identifier > 11 for identifier in parsed):
                raise ValueError("Group identifiers must be between 1 and 11")
            if 12 in parsed:
                raise ValueError("Group list must not include 12 (Other)")
            allowed_groups = parsed
        else:
            tokens.append(token)
    if len(tokens) not in (3, 4):
        raise ValueError(
            "Unsupported argument format; expected DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS] with optional group=...",
        )
    volume_filter, buy_strategy_name, sell_strategy_name = tokens[:3]
    stop_loss_percentage = float(tokens[3]) if len(tokens) == 4 else 1.0
    minimum_average_dollar_volume: float | None = None
    minimum_average_dollar_volume_ratio: float | None = None
    top_dollar_volume_rank: int | None = None
    # Accept both TopN and legacy Nth; case-insensitive for Top
    combined_percentage_top_match = re.fullmatch(
        r"dollar_volume>(\d+(?:\.\d{1,2})?)%,Top(\d+)",
        volume_filter,
        flags=re.IGNORECASE,
    )
    combined_percentage_nth_match = re.fullmatch(
        r"dollar_volume>(\d+(?:\.\d{1,2})?)%,(\d+)th",
        volume_filter,
    )
    if combined_percentage_top_match is not None or combined_percentage_nth_match is not None:
        match_obj = combined_percentage_top_match or combined_percentage_nth_match
        minimum_average_dollar_volume_ratio = float(match_obj.group(1)) / 100
        top_dollar_volume_rank = int(match_obj.group(2))
    else:
        combined_top_match = re.fullmatch(
            r"dollar_volume>(\d+(?:\.\d+)?),Top(\d+)",
            volume_filter,
            flags=re.IGNORECASE,
        )
        combined_nth_match = re.fullmatch(
            r"dollar_volume>(\d+(?:\.\d+)?),(\d+)th",
            volume_filter,
        )
        if combined_top_match is not None or combined_nth_match is not None:
            match_obj = combined_top_match or combined_nth_match
            minimum_average_dollar_volume = float(match_obj.group(1))
            top_dollar_volume_rank = int(match_obj.group(2))
        else:
            percentage_match = re.fullmatch(
                r"dollar_volume>(\d+(?:\.\d{1,2})?)%",
                volume_filter,
            )
            if percentage_match is not None:
                minimum_average_dollar_volume_ratio = float(percentage_match.group(1)) / 100
            else:
                volume_match = re.fullmatch(
                    r"dollar_volume>(\d+(?:\.\d+)?)",
                    volume_filter,
                )
                if volume_match is not None:
                    minimum_average_dollar_volume = float(volume_match.group(1))
                else:
                    rank_top_match = re.fullmatch(
                        r"dollar_volume=Top(\d+)",
                        volume_filter,
                        flags=re.IGNORECASE,
                    )
                    rank_nth_match = re.fullmatch(
                        r"dollar_volume=(\d+)th",
                        volume_filter,
                    )
                    if rank_top_match is not None or rank_nth_match is not None:
                        top_dollar_volume_rank = int((rank_top_match or rank_nth_match).group(1))
                    else:
                        raise ValueError(
                            "Unsupported filter format. Expected 'dollar_volume>NUMBER', "
                            "'dollar_volume>NUMBER%', 'dollar_volume=TopN' (or 'Nth'), "
                            "'dollar_volume>NUMBER,TopN' (or ',Nth'), or "
                            "'dollar_volume>NUMBER%,TopN' (or ',Nth').",
                        )
    return (
        minimum_average_dollar_volume_ratio
        if minimum_average_dollar_volume_ratio is not None
        else minimum_average_dollar_volume,
        top_dollar_volume_rank,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
        allowed_groups,
    )


def run_daily_tasks(
    buy_strategy_name: str,
    sell_strategy_name: str,
    start_date: str,
    end_date: str,
    symbol_list: Iterable[str] | None = None,
    data_download_function: Callable[[str, str, str], pandas.DataFrame] = download_history,
    data_directory: Path | None = None,
    minimum_average_dollar_volume: float | None = None,
    top_dollar_volume_rank: int | None = None,  # TODO: review
    allowed_fama_french_groups: set[int] | None = None,
) -> Dict[str, List[str]]:
    """Execute the daily workflow for data retrieval and signal detection.

    Parameters
    ----------
    buy_strategy_name: str
        Name of the strategy providing entry signals.
    sell_strategy_name: str
        Name of the strategy providing exit signals.
    start_date: str
        Start date for downloading historical data in ``YYYY-MM-DD`` format.
    end_date: str
        End date for downloading historical data in ``YYYY-MM-DD`` format.
    symbol_list: Iterable[str] | None
        Iterable of ticker symbols to process. When ``None``, the local symbol
        cache is updated and used.
    data_download_function: Callable[[str, str, str], pandas.DataFrame]
        Function responsible for retrieving historical price data. Defaults to
        :func:`download_history`.
    data_directory: Path | None
        Optional directory path where downloaded data is stored as CSV files.
    minimum_average_dollar_volume: float | None
        Minimum 50-day average dollar volume in millions required for a symbol
        to be processed. When ``None``, no volume filter is applied.
    top_dollar_volume_rank: int | None
        When provided, only the ``N`` symbols with the highest 50-day average
        dollar volume are processed.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with ``entry_signals`` and ``exit_signals`` listing symbols
        that triggered the respective signals on the latest available data row.
    """
    try:
        update_symbol_cache()
    except Exception as update_error:  # noqa: BLE001
        LOGGER.warning("Could not update symbol cache: %s", update_error)
    if symbol_list is None:
        # Require a YF-verified symbol universe; do not fall back to SEC list
        symbol_list = load_yf_symbols()
        if not symbol_list:
            raise ValueError(
                "No Yahoo Finance symbol list available. Run 'update_yf_symbols' first."
            )
    # Apply FF12 group filtering if sector data is available.
    try:
        symbol_to_group = load_ff12_groups_by_symbol()
    except Exception:  # noqa: BLE001
        symbol_to_group = {}
    if symbol_to_group:
        # Always exclude 'Other' (12)
        filtered_symbols: list[str] = []
        for ticker in symbol_list:
            group_id = symbol_to_group.get(ticker.upper())
            if group_id is None:
                if allowed_fama_french_groups is None:
                    # When not restricting groups, keep symbols with unknown mapping
                    filtered_symbols.append(ticker)
                else:
                    # When restricting, drop unknowns to be safe
                    continue
            else:
                if group_id == 12:
                    continue
                if (
                    allowed_fama_french_groups is None
                    or group_id in allowed_fama_french_groups
                ):
                    filtered_symbols.append(ticker)
        symbol_list = filtered_symbols

    entry_signal_symbols: List[str] = []
    exit_signal_symbols: List[str] = []

    # Allow parameterized strategy names by parsing out the base name.
    from .strategy import (
        BUY_STRATEGIES,
        SELL_STRATEGIES,
        parse_strategy_name,
        _extract_sma_factor,
        _extract_short_long_windows_for_20_50,
    )

    try:
        buy_base_name, buy_window_size, buy_slope_range = parse_strategy_name(
            buy_strategy_name
        )
    except Exception as parse_error:  # noqa: BLE001
        raise ValueError(f"Unknown strategy: {buy_strategy_name}") from parse_error
    try:
        sell_base_name, sell_window_size, sell_slope_range = parse_strategy_name(
            sell_strategy_name
        )
    except Exception as parse_error:  # noqa: BLE001
        raise ValueError(f"Unknown strategy: {sell_strategy_name}") from parse_error

    if buy_base_name not in BUY_STRATEGIES:
        raise ValueError(f"Unknown strategy: {buy_strategy_name}")
    if sell_base_name not in SELL_STRATEGIES:
        raise ValueError(f"Unknown strategy: {sell_strategy_name}")

    symbol_data: List[tuple[str, pandas.DataFrame, float | None]] = []
    for symbol in symbol_list:
        data_file_path: Path | None = None
        if data_directory is not None:
            data_directory.mkdir(parents=True, exist_ok=True)
            data_file_path = data_directory / f"{symbol}.csv"
        try:
            if data_file_path is not None:
                price_history_frame = data_download_function(
                    symbol, start_date, end_date, cache_path=data_file_path
                )
            else:
                price_history_frame = data_download_function(symbol, start_date, end_date)
        except Exception as download_error:  # noqa: BLE001
            LOGGER.warning("Failed to download data for %s: %s", symbol, download_error)
            continue
        if price_history_frame.empty:
            LOGGER.warning("No data returned for %s", symbol)
            continue

        average_dollar_volume: float | None = None
        if "volume" in price_history_frame.columns:
            dollar_volume_series = price_history_frame["close"] * price_history_frame["volume"]
            if not dollar_volume_series.empty:
                average_dollar_volume = float(
                    dollar_volume_series.rolling(window=50).mean().iloc[-1]
                )
        symbol_data.append((symbol, price_history_frame, average_dollar_volume))

    if minimum_average_dollar_volume is not None:
        symbol_data = [
            item
            for item in symbol_data
            if item[2] is not None
            and (item[2] / 1_000_000) >= minimum_average_dollar_volume
        ]

    if top_dollar_volume_rank is not None:
        symbol_data = [item for item in symbol_data if item[2] is not None]
        symbol_data.sort(key=lambda item: item[2], reverse=True)
        symbol_data = symbol_data[:top_dollar_volume_rank]

    def _apply_strategy(
        full_name: str,
        base_name: str,
        window_size: int | None,
        slope_range: tuple[float, float] | None,
        table,
        frame: pandas.DataFrame,
    ) -> None:
        kwargs: dict = {}
        if base_name == "20_50_sma_cross":
            maybe_windows = _extract_short_long_windows_for_20_50(full_name)
            if maybe_windows is not None:
                kwargs["short_window_size"], kwargs["long_window_size"] = maybe_windows
        else:
            if window_size is not None:
                kwargs["window_size"] = window_size
            if slope_range is not None:
                kwargs["slope_range"] = slope_range
            sma_factor_value = _extract_sma_factor(full_name)
            if sma_factor_value is not None and base_name in {"ema_sma_cross", "ema_sma_cross_with_slope"}:
                kwargs["sma_window_factor"] = sma_factor_value
        table[base_name](frame, **kwargs)
        if base_name != full_name:
            frame.rename(
                columns={
                    f"{base_name}_entry_signal": f"{full_name}_entry_signal",
                    f"{base_name}_exit_signal": f"{full_name}_exit_signal",
                },
                inplace=True,
            )

    for symbol, price_history_frame, _ in symbol_data:
        _apply_strategy(
            buy_strategy_name,
            buy_base_name,
            buy_window_size,
            buy_slope_range,
            BUY_STRATEGIES,
            price_history_frame,
        )
        if buy_strategy_name != sell_strategy_name:
            _apply_strategy(
                sell_strategy_name,
                sell_base_name,
                sell_window_size,
                sell_slope_range,
                SELL_STRATEGIES,
                price_history_frame,
            )

        entry_column_name = f"{buy_strategy_name}_entry_signal"
        exit_column_name = f"{sell_strategy_name}_exit_signal"
        latest_row = price_history_frame.iloc[-1]
        if entry_column_name in price_history_frame and bool(latest_row[entry_column_name]):
            entry_signal_symbols.append(symbol)
        if exit_column_name in price_history_frame and bool(latest_row[exit_column_name]):
            exit_signal_symbols.append(symbol)

    return {"entry_signals": entry_signal_symbols, "exit_signals": exit_signal_symbols}


def run_daily_tasks_from_argument(
    argument_line: str,
    start_date: str,
    end_date: str,
    symbol_list: Iterable[str] | None = None,
    data_download_function: Callable[[str, str, str], pandas.DataFrame] = download_history,
    data_directory: Path | None = None,
) -> Dict[str, List[str]]:
    """Run daily tasks using a single argument string.

    Parameters
    ----------
    argument_line: str
        Argument string in the format accepted by
        :func:`parse_daily_task_arguments`.
    start_date: str
        Start date for downloading historical data in ``YYYY-MM-DD`` format.
    end_date: str
        End date for downloading historical data in ``YYYY-MM-DD`` format.
    symbol_list: Iterable[str] | None
        Iterable of ticker symbols to process. When ``None``, the local symbol
        cache is updated and used.
    data_download_function: Callable[[str, str, str], pandas.DataFrame]
        Function responsible for retrieving historical price data. Defaults to
        :func:`download_history`.
    data_directory: Path | None
        Optional directory path where downloaded data is stored as CSV files.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with ``entry_signals`` and ``exit_signals`` listing symbols
        that triggered the respective signals on the latest available data row.
    """
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        buy_strategy_name,
        sell_strategy_name,
        _,
        allowed_groups,
    ) = parse_daily_task_arguments(argument_line)
    return run_daily_tasks(
        buy_strategy_name=buy_strategy_name,
        sell_strategy_name=sell_strategy_name,
        start_date=start_date,
        end_date=end_date,
        symbol_list=symbol_list,
        data_download_function=data_download_function,
        data_directory=data_directory,
        minimum_average_dollar_volume=minimum_average_dollar_volume,
        top_dollar_volume_rank=top_dollar_volume_rank,
        allowed_fama_french_groups=allowed_groups,
    )
