"""Scheduled daily tasks for updating data and evaluating strategies."""
# TODO: review

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import pandas

from .symbols import update_symbol_cache, load_symbols
from .data_loader import download_history
from .strategy import SUPPORTED_STRATEGIES

LOGGER = logging.getLogger(__name__)


def run_daily_tasks(
    strategy_name: str,
    start_date: str,
    end_date: str,
    symbol_list: Iterable[str] | None = None,
    data_download_function: Callable[[str, str, str], pandas.DataFrame] = download_history,
    data_directory: Path | None = None,
) -> Dict[str, List[str]]:
    """Execute the daily workflow for data retrieval and signal detection.

    Parameters
    ----------
    strategy_name: str
        Name of the strategy defined in :data:`SUPPORTED_STRATEGIES`.
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
    update_symbol_cache()
    if symbol_list is None:
        symbol_list = load_symbols()

    entry_signal_symbols: List[str] = []
    exit_signal_symbols: List[str] = []

    if strategy_name not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    strategy_function = SUPPORTED_STRATEGIES[strategy_name]

    for symbol in symbol_list:
        try:
            price_history_frame = data_download_function(symbol, start_date, end_date)
        except Exception as download_error:  # noqa: BLE001
            LOGGER.warning("Failed to download data for %s: %s", symbol, download_error)
            continue
        if price_history_frame.empty:
            LOGGER.warning("No data returned for %s", symbol)
            continue

        strategy_function(price_history_frame)
        entry_column_name = f"{strategy_name}_entry_signal"
        exit_column_name = f"{strategy_name}_exit_signal"
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
