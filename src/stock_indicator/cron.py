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
from .strategy import SUPPORTED_STRATEGIES

LOGGER = logging.getLogger(__name__)


def parse_daily_task_arguments(argument_line: str) -> Tuple[
    float | None,
    int | None,
    str,
    str,
    float,
]:
    """Parse a cron job argument string.

    The expected format is ``dollar_volume>NUMBER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``,
    ``dollar_volume>NUMBER% BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``,
    ``dollar_volume=Nth BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``,
    ``dollar_volume>NUMBER,Nth BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``, or
    ``dollar_volume>NUMBER%,Nth BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``.

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
    if len(argument_parts) not in (3, 4):
        raise ValueError(
            "argument_line must be 'dollar_volume>NUMBER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', "
            "'dollar_volume>NUMBER% BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', "
            "'dollar_volume=RANKth BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', "
            "'dollar_volume>NUMBER,RANKth BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', or "
            "'dollar_volume>NUMBER%,RANKth BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]'",
        )
    volume_filter, buy_strategy_name, sell_strategy_name = argument_parts[:3]
    stop_loss_percentage = (
        float(argument_parts[3]) if len(argument_parts) == 4 else 1.0
    )
    minimum_average_dollar_volume: float | None = None
    minimum_average_dollar_volume_ratio: float | None = None
    top_dollar_volume_rank: int | None = None
    combined_percentage_match = re.fullmatch(
        r"dollar_volume>(\d+(?:\.\d{1,2})?)%,(\d+)th",
        volume_filter,
    )
    if combined_percentage_match is not None:
        minimum_average_dollar_volume_ratio = (
            float(combined_percentage_match.group(1)) / 100
        )
        top_dollar_volume_rank = int(combined_percentage_match.group(2))
    else:
        combined_match = re.fullmatch(
            r"dollar_volume>(\d+(?:\.\d+)?),(\d+)th",
            volume_filter,
        )
        if combined_match is not None:
            minimum_average_dollar_volume = float(combined_match.group(1))
            top_dollar_volume_rank = int(combined_match.group(2))
        else:
            percentage_match = re.fullmatch(
                r"dollar_volume>(\d+(?:\.\d{1,2})?)%",
                volume_filter,
            )
            if percentage_match is not None:
                minimum_average_dollar_volume_ratio = (
                    float(percentage_match.group(1)) / 100
                )
            else:
                volume_match = re.fullmatch(
                    r"dollar_volume>(\d+(?:\.\d+)?)",
                    volume_filter,
                )
                if volume_match is not None:
                    minimum_average_dollar_volume = float(volume_match.group(1))
                else:
                    rank_match = re.fullmatch(
                        r"dollar_volume=(\d+)th",
                        volume_filter,
                    )
                    if rank_match is not None:
                        top_dollar_volume_rank = int(rank_match.group(1))
                    else:
                        raise ValueError(
                            "Unsupported filter format. Expected 'dollar_volume>NUMBER', "
                            "'dollar_volume>NUMBER%', 'dollar_volume=RANKth', "
                            "'dollar_volume>NUMBER,RANKth', or "
                            "'dollar_volume>NUMBER%,RANKth'.",
                        )
    return (
        minimum_average_dollar_volume_ratio
        if minimum_average_dollar_volume_ratio is not None
        else minimum_average_dollar_volume,
        top_dollar_volume_rank,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
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

    entry_signal_symbols: List[str] = []
    exit_signal_symbols: List[str] = []

    if buy_strategy_name not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown strategy: {buy_strategy_name}")
    if sell_strategy_name not in SUPPORTED_STRATEGIES:
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

    for symbol, price_history_frame, _ in symbol_data:
        SUPPORTED_STRATEGIES[buy_strategy_name](price_history_frame)
        if buy_strategy_name != sell_strategy_name:
            SUPPORTED_STRATEGIES[sell_strategy_name](price_history_frame)

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
    )
