"""Helper functions for managing historical data and signals."""
# TODO: review

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

import pandas
from pandas.tseries.offsets import BDay

from .cron import parse_daily_task_arguments, run_daily_tasks
from .data_loader import download_history, load_local_history
from .symbols import SP500_SYMBOL, load_symbols
from . import strategy

LOGGER = logging.getLogger(__name__)

DEFAULT_START_DATE = "2019-01-01"
# Earliest date used when refreshing historical data; this guards against
# missing rows in local caches. Modify only when extending the supported
# history range.
MINIMUM_HISTORY_DATE = "2014-01-01"
DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
STOCK_DATA_DIRECTORY = DATA_DIRECTORY / "stock_data"


def determine_latest_trading_date(
    now: datetime.datetime | None = None,
) -> datetime.date:
    """Return the most recent trading date based on Eastern Time.

    Parameters
    ----------
    now:
        Current timestamp. When ``None`` the system time in the ``US/Eastern``
        timezone is used.

    Returns
    -------
    datetime.date
        The prior business day if the time is earlier than 16:00 Eastern,
        otherwise the current date.
    """

    eastern_zone = ZoneInfo("US/Eastern")
    if now is None:
        current_time = datetime.datetime.now(tz=eastern_zone)
    else:
        current_time = (
            now.astimezone(eastern_zone)
            if now.tzinfo is not None
            else now.replace(tzinfo=eastern_zone)
        )

    if current_time.time() < datetime.time(16, 0):
        previous_business_day = (
            pandas.Timestamp(current_time.date()) - BDay(1)
        ).date()
        return previous_business_day
    return current_time.date()


def determine_start_date(data_directory: Path) -> str:
    """Return the earliest date across all CSV files in ``data_directory``.

    When no CSV files are available, ``DEFAULT_START_DATE`` is returned.
    """

    earliest_date: datetime.date | None = None
    if not data_directory.exists():
        return DEFAULT_START_DATE
    for csv_file_path in data_directory.glob("*.csv"):
        try:
            date_frame = pandas.read_csv(
                csv_file_path, usecols=[0], parse_dates=[0]
            )
        except Exception as read_error:  # noqa: BLE001
            LOGGER.warning("Could not read %s: %s", csv_file_path, read_error)
            continue
        if date_frame.empty:
            continue
        column_minimum = date_frame.iloc[:, 0].min()
        if not hasattr(column_minimum, "date"):
            continue
        earliest_candidate = column_minimum.date()
        if earliest_date is None or earliest_candidate < earliest_date:
            earliest_date = earliest_candidate
    if earliest_date is None:
        return DEFAULT_START_DATE
    return earliest_date.isoformat()


def determine_last_cached_date(data_directory: Path) -> datetime.date:
    """Return the most recent date found in any CSV under ``data_directory``."""

    latest_date: datetime.date | None = None
    if data_directory.exists():
        for csv_file_path in data_directory.glob("*.csv"):
            try:
                date_frame = pandas.read_csv(
                    csv_file_path, usecols=[0], parse_dates=[0]
                )
            except Exception as read_error:  # noqa: BLE001
                LOGGER.warning("Could not read %s: %s", csv_file_path, read_error)
                continue
            if date_frame.empty:
                continue
            value = date_frame.iloc[-1, 0]
            if hasattr(value, "date"):
                current_date = value.date()
                if latest_date is None or current_date > latest_date:
                    latest_date = current_date
    if latest_date is None:
        return datetime.date.fromisoformat(DEFAULT_START_DATE)
    return latest_date


def update_all_data_from_yf(
    start_date: str, end_date: str, data_directory: Path
) -> None:
    """Download historical data for all symbols into ``data_directory``.

    The ``end_date`` argument is treated as inclusive. To accommodate the
    exclusive end-date semantics of the Yahoo Finance API, this function adds
    one day to ``end_date`` before requesting data.
    """

    exclusive_end_date = (
        datetime.date.fromisoformat(end_date) + datetime.timedelta(days=1)
    ).isoformat()
    symbol_list = load_symbols()
    if SP500_SYMBOL not in symbol_list:
        symbol_list.append(SP500_SYMBOL)
    for symbol_name in symbol_list:
        csv_path = data_directory / f"{symbol_name}.csv"
        try:
            download_history(
                symbol_name,
                start=start_date,
                end=exclusive_end_date,
                cache_path=csv_path,
            )
            try:
                cached_frame = pandas.read_csv(
                    csv_path, index_col=0, parse_dates=True
                )
                deduplicated_frame = cached_frame.loc[
                    ~cached_frame.index.duplicated(keep="last")
                ]
                deduplicated_frame.to_csv(csv_path)
            except Exception as cache_error:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to deduplicate %s: %s", csv_path, cache_error
                )
        except Exception as download_error:  # noqa: BLE001
            LOGGER.warning(
                "Failed to refresh data for %s: %s", symbol_name, download_error
            )


def find_history_signal(
    date_string: str | None,
    dollar_volume_filter: str,
    buy_strategy: str,
    sell_strategy: str,
    stop_loss: float,
    allowed_fama_french_groups: set[int] | None = None,
) -> Dict[str, List[str]]:
    """Find entry and exit signals for a single historical date.

    When ``date_string`` is ``None`` the most recent trading date is determined
    via :func:`determine_latest_trading_date` and used for evaluation. Entries
    based on generated signals occur on the next trading day's open.

    Parameters
    ----------
    date_string:
        ISO formatted date string representing the signal date. ``None``
        triggers evaluation for the latest trading day.
    dollar_volume_filter:
        Filter applied to select symbols based on dollar volume.
    buy_strategy:
        Name of the strategy used to generate entry signals.
    sell_strategy:
        Name of the strategy used to generate exit signals.
    stop_loss:
        Fractional loss used for downstream simulations; not used in signal
        detection here but preserved for parity with other entry points.
    allowed_fama_french_groups:
        Optional set of FF12 group identifiers (1–11) used to restrict the
        tradable universe. Group 12 (Other) is always excluded when sector data
        is available.

    Historical data starting from either the earliest cached date in
    ``data/stock_data`` or ``2014-01-01``—whichever is earlier—is used to
    ensure sufficient look-back.

    Returns
    -------
    Dict[str, List[str] | List[tuple[str, int | None]]]
        Dictionary containing ``filtered_symbols`` (pairs of symbol and
        Fama–French group identifiers), ``entry_signals`` and ``exit_signals``.
    """

    # TODO: review
    if date_string is None:
        date_string = determine_latest_trading_date().isoformat()
    group_token = (
        ""
        if not allowed_fama_french_groups
        else "group=" + ",".join(str(i) for i in sorted(allowed_fama_french_groups)) + " "
    )
    argument_line = f"{group_token}{dollar_volume_filter} {buy_strategy} {sell_strategy} {stop_loss}"
    start_date_string = determine_start_date(STOCK_DATA_DIRECTORY)
    minimum_timestamp = pandas.Timestamp(MINIMUM_HISTORY_DATE)
    if pandas.Timestamp(start_date_string) > minimum_timestamp:
        start_date_string = MINIMUM_HISTORY_DATE
    try:
        evaluation_timestamp = pandas.Timestamp(date_string)
        evaluation_end_date_string = evaluation_timestamp.date().isoformat()
    except Exception:  # noqa: BLE001
        evaluation_timestamp = pandas.Timestamp.today()
        evaluation_end_date_string = evaluation_timestamp.date().isoformat()
    try:
        local_symbols = [
            csv_path.stem
            for csv_path in STOCK_DATA_DIRECTORY.glob("*.csv")
            if csv_path.stem and csv_path.stem != "^GSPC"
        ]
    except Exception:  # noqa: BLE001
        local_symbols = None
    if local_symbols is not None:
        missing_symbols: List[str] = []
        for symbol_name in local_symbols:
            csv_file_path = STOCK_DATA_DIRECTORY / f"{symbol_name}.csv"
            try:
                history_frame = pandas.read_csv(
                    csv_file_path, index_col=0, parse_dates=True
                )
            except Exception:  # noqa: BLE001
                missing_symbols.append(symbol_name)
                continue
            if evaluation_timestamp not in history_frame.index:
                missing_symbols.append(symbol_name)
        if missing_symbols:
            warning_symbol_list = ", ".join(sorted(missing_symbols))
            LOGGER.warning(
                "Skipping symbols missing %s: %s",
                date_string,
                warning_symbol_list,
            )
            local_symbols = [
                symbol_name
                for symbol_name in local_symbols
                if symbol_name not in missing_symbols
            ]
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
        parsed_buy_strategy,
        parsed_sell_strategy,
        _,
        allowed_groups,
    ) = parse_daily_task_arguments(argument_line)
    signal_result: Dict[str, List[str] | List[tuple[str, int | None]]] = run_daily_tasks(
        buy_strategy_name=parsed_buy_strategy,
        sell_strategy_name=parsed_sell_strategy,
        start_date=start_date_string,
        end_date=evaluation_end_date_string,
        symbol_list=local_symbols,
        data_download_function=load_local_history,
        data_directory=STOCK_DATA_DIRECTORY,
        minimum_average_dollar_volume=minimum_average_dollar_volume,
        top_dollar_volume_rank=top_dollar_volume_rank,
        allowed_fama_french_groups=allowed_groups,
        maximum_symbols_per_group=maximum_symbols_per_group,
        use_unshifted_signals=True,
    )
    entry_signals = signal_result.get("entry_signals", [])
    exit_signals = signal_result.get("exit_signals", [])
    filtered_symbols = signal_result.get("filtered_symbols", [])
    return {
        "filtered_symbols": filtered_symbols,
        "entry_signals": entry_signals,
        "exit_signals": exit_signals,
    }


def filter_debug_values(
    symbol_name: str,
    evaluation_date_string: str,
    buy_strategy_name: str,
    sell_strategy_name: str,
) -> Dict[str, float | bool | None]:
    """Return indicator debug values for ``symbol_name`` on ``evaluation_date_string``.

    Loads local price history, attaches indicators for the provided buy and sell
    strategies, and extracts a handful of useful columns for debugging
    threshold-based filters.
    """

    # TODO: review
    csv_file_path = STOCK_DATA_DIRECTORY / f"{symbol_name}.csv"
    start_date_string = determine_start_date(STOCK_DATA_DIRECTORY)
    end_date_string = (
        pandas.Timestamp(evaluation_date_string) + pandas.Timedelta(days=1)
    ).date().isoformat()
    price_history_frame = load_local_history(
        symbol_name, start_date_string, end_date_string, cache_path=csv_file_path
    )
    if price_history_frame.empty or pandas.Timestamp(evaluation_date_string) not in price_history_frame.index:
        return {
            "sma_angle": None,
            "near_price_volume_ratio": None,
            "above_price_volume_ratio": None,
            "entry": False,
            "exit": False,
        }

    (
        buy_base_name,
        buy_window_size,
        buy_angle_range,
        buy_near_percentage,
        buy_above_percentage,
    ) = strategy.parse_strategy_name(buy_strategy_name)
    buy_function = strategy.BUY_STRATEGIES.get(buy_base_name)
    if buy_function is not None:
        buy_arguments: Dict[str, float | tuple[float, float]] = {}
        if buy_window_size is not None:
            buy_arguments["window_size"] = buy_window_size
        if buy_angle_range is not None:
            buy_arguments["angle_range"] = buy_angle_range
        if buy_near_percentage is not None:
            buy_arguments["near_pct"] = buy_near_percentage
        if buy_above_percentage is not None:
            buy_arguments["above_pct"] = buy_above_percentage
        buy_function(price_history_frame, **buy_arguments)

    (
        sell_base_name,
        sell_window_size,
        sell_angle_range,
        sell_near_percentage,
        sell_above_percentage,
    ) = strategy.parse_strategy_name(sell_strategy_name)
    sell_function = strategy.SELL_STRATEGIES.get(sell_base_name)
    if sell_function is not None:
        sell_arguments: Dict[str, float | tuple[float, float]] = {}
        if sell_window_size is not None:
            sell_arguments["window_size"] = sell_window_size
        if sell_angle_range is not None:
            sell_arguments["angle_range"] = sell_angle_range
        if sell_near_percentage is not None:
            sell_arguments["near_pct"] = sell_near_percentage
        if sell_above_percentage is not None:
            sell_arguments["above_pct"] = sell_above_percentage
        sell_function(price_history_frame, **sell_arguments)

    row = price_history_frame.loc[pandas.Timestamp(evaluation_date_string)]
    return {
        "sma_angle": row.get("sma_angle"),
        "near_price_volume_ratio": row.get("near_price_volume_ratio"),
        "above_price_volume_ratio": row.get("above_price_volume_ratio"),
        "entry": bool(row.get(f"{buy_base_name}_entry_signal", False)),
        "exit": bool(row.get(f"{sell_base_name}_exit_signal", False)),
    }

