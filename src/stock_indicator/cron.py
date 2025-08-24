"""Scheduled daily tasks for updating data and evaluating strategies."""
# TODO: review

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas

from .symbols import update_symbol_cache, load_symbols
from .data_loader import download_history
from .strategy import SUPPORTED_STRATEGIES

LOGGER = logging.getLogger(__name__)


def parse_daily_task_arguments(argument_line: str) -> Tuple[float, str, str, float]:
    """Parse a cron job argument string.

    The expected format is ``dollar_volume>NUMBER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``.

    Parameters
    ----------
    argument_line: str
        Argument string describing filters and strategy names.

    Returns
    -------
    Tuple[float, str, str, float]
        Tuple containing minimum dollar volume, buy strategy name, sell strategy
        name, and stop loss percentage.
    """
    argument_parts = argument_line.split()
    if len(argument_parts) not in (3, 4):
        raise ValueError(
            "argument_line must be of the form 'dollar_volume>NUMBER BUY_STRATEGY "
            "SELL_STRATEGY [STOP_LOSS]'"
        )
    volume_filter, buy_strategy_name, sell_strategy_name = argument_parts[:3]
    stop_loss_percentage = float(argument_parts[3]) if len(argument_parts) == 4 else 1.0
    volume_match = re.fullmatch(r"dollar_volume>(\d+(?:\.\d+)?)", volume_filter)
    if volume_match is None:
        raise ValueError("Unsupported filter format")
    minimum_average_dollar_volume = float(volume_match.group(1))
    return (
        minimum_average_dollar_volume,
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
        symbol_list = load_symbols()

    entry_signal_symbols: List[str] = []
    exit_signal_symbols: List[str] = []

    if buy_strategy_name not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown strategy: {buy_strategy_name}")
    if sell_strategy_name not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown strategy: {sell_strategy_name}")

    for symbol in symbol_list:
        try:
            price_history_frame = data_download_function(symbol, start_date, end_date)
        except Exception as download_error:  # noqa: BLE001
            LOGGER.warning("Failed to download data for %s: %s", symbol, download_error)
            continue
        if price_history_frame.empty:
            LOGGER.warning("No data returned for %s", symbol)
            continue

        if minimum_average_dollar_volume is not None:
            if "volume" not in price_history_frame.columns:
                LOGGER.warning("Volume column is missing for %s", symbol)
                continue
            dollar_volume_series = price_history_frame["close"] * price_history_frame["volume"]
            recent_average_dollar_volume = (
                dollar_volume_series.rolling(window=50).mean().iloc[-1] / 1_000_000
            )
            if pandas.isna(recent_average_dollar_volume) or (
                recent_average_dollar_volume < minimum_average_dollar_volume
            ):
                continue

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

        if data_directory is not None:
            data_directory.mkdir(parents=True, exist_ok=True)
            data_file_path = data_directory / f"{symbol}.csv"
            price_history_frame.to_csv(data_file_path)

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
    )
