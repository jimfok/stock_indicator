# TODO: review
"""Run named strategy-set backtests without the interactive management shell."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import pandas

from stock_indicator import manage, strategy
from stock_indicator.daily_job import determine_start_date
from stock_indicator.strategy_sets import load_strategy_set_mapping

LOGGER = logging.getLogger(__name__)
DEFAULT_START_DATE = pandas.Timestamp("1990-01-01")
DEFAULT_LOG_PATH = Path("logs/strategy_backtest.log")


def configure_logging(log_path: Path) -> None:
    """Configure logging to emit to stdout and the specified log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def parse_strategy_identifiers(raw_values: Sequence[str]) -> List[str]:
    """Normalize and deduplicate strategy identifiers while preserving order."""
    normalized_identifiers: List[str] = []
    seen_identifiers: Set[str] = set()
    for raw_identifier in raw_values:
        identifier = raw_identifier.strip().lower()
        if not identifier or identifier in seen_identifiers:
            continue
        normalized_identifiers.append(identifier)
        seen_identifiers.add(identifier)
    return normalized_identifiers


def parse_group_identifiers(raw_value: str | None) -> Set[int] | None:
    """Parse a comma-separated list of Fama–French group identifiers."""
    if raw_value is None:
        return None
    identifiers: Set[int] = set()
    for segment in raw_value.split(","):
        stripped_segment = segment.strip()
        if not stripped_segment:
            continue
        try:
            identifier_value = int(stripped_segment)
        except ValueError as error:
            raise ValueError(
                f"Invalid group identifier '{stripped_segment}'"
            ) from error
        if identifier_value < 1 or identifier_value > 11:
            raise ValueError(
                "Group identifiers must be between 1 and 11 (inclusive); "
                f"received {identifier_value}"
            )
        identifiers.add(identifier_value)
    return identifiers if identifiers else None


def format_summary_line(label: str, metrics: strategy.StrategyMetrics) -> str:
    """Format the key performance metrics for display."""
    return (
        f"[{label}] Trades: {metrics.total_trades}, "
        f"Win rate: {metrics.win_rate:.2%}, "
        f"Mean profit %: {metrics.mean_profit_percentage:.2%}, "
        f"Profit % Std Dev: {metrics.profit_percentage_standard_deviation:.2%}, "
        f"Mean loss %: {metrics.mean_loss_percentage:.2%}, "
        f"Loss % Std Dev: {metrics.loss_percentage_standard_deviation:.2%}, "
        f"Mean holding period: {metrics.mean_holding_period:.2f} bars, "
        f"Holding period Std Dev: {metrics.holding_period_standard_deviation:.2f} bars, "
        f"Max concurrent positions: {metrics.maximum_concurrent_positions}, "
        f"Final balance: {metrics.final_balance:.2f}, "
        f"CAGR: {metrics.compound_annual_growth_rate:.2%}, "
        f"Max drawdown: {metrics.maximum_drawdown:.2%}"
    )


def clean_trade_details(metrics: strategy.StrategyMetrics) -> None:
    """Remove pre-2014 GOOGL trades to align with management shell output."""
    earliest_valid_googl_date = pandas.Timestamp("2014-04-03")
    filtered_trade_details_by_year: dict[int, list[strategy.TradeDetail]] = {}
    removed_any_trade = False
    for year_value, detail_list in metrics.trade_details_by_year.items():
        retained_details: list[strategy.TradeDetail] = []
        for trade_detail in detail_list:
            if (
                trade_detail.symbol == "GOOGL"
                and trade_detail.date < earliest_valid_googl_date
            ):
                removed_any_trade = True
                continue
            retained_details.append(trade_detail)
        if retained_details:
            filtered_trade_details_by_year[year_value] = retained_details
    if removed_any_trade:
        metrics.trade_details_by_year = filtered_trade_details_by_year


def display_trade_details(
    label: str,
    metrics: strategy.StrategyMetrics,
) -> None:
    """Log trade details grouped by year."""
    if not metrics.trade_details_by_year:
        LOGGER.info("[%s] No trade details available.", label)
        return
    for year_value in sorted(metrics.trade_details_by_year):
        LOGGER.info("[%s] Year %s:", label, year_value)
        for trade_detail in metrics.trade_details_by_year[year_value]:
            LOGGER.info(
                "[%s]   %s %s %s %s result=%s change=%s reason=%s",
                label,
                trade_detail.date.date(),
                trade_detail.symbol,
                trade_detail.action.upper(),
                f"{trade_detail.price:.2f}",
                trade_detail.result,
                (
                    f"{trade_detail.percentage_change:.2%}"
                    if trade_detail.percentage_change is not None
                    else "N/A"
                ),
                trade_detail.exit_reason,
            )


def run_strategy_set(
    strategy_identifier: str,
    *,
    data_directory: Path,
    volume_filter: str,
    starting_cash: float,
    withdraw_amount: float,
    stop_loss_percentage: float,
    start_date: pandas.Timestamp,
    margin_multiplier: float,
    allowed_groups: Set[int] | None,
    show_details: bool,
) -> None:
    """Execute a strategy-set backtest and log the summary."""
    strategy_mapping = load_strategy_set_mapping()
    if strategy_identifier not in strategy_mapping:
        raise ValueError(f"Unknown strategy id: {strategy_identifier}")
    buy_strategy_name, sell_strategy_name = strategy_mapping[strategy_identifier]
    if not manage._has_supported_strategy(  # type: ignore[attr-defined]
        buy_strategy_name, strategy.BUY_STRATEGIES
    ):
        raise ValueError(
            f"Unsupported buy strategy '{buy_strategy_name}' for {strategy_identifier}"
        )
    if not manage._has_supported_strategy(  # type: ignore[attr-defined]
        sell_strategy_name, strategy.SELL_STRATEGIES
    ):
        raise ValueError(
            f"Unsupported sell strategy '{sell_strategy_name}' for {strategy_identifier}"
        )
    (
        minimum_average_dollar_volume,
        minimum_average_dollar_volume_ratio,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
    ) = manage._parse_volume_filter(volume_filter)  # type: ignore[attr-defined]
    extra_arguments: dict[str, object] = {}
    if maximum_symbols_per_group != 1:
        extra_arguments["maximum_symbols_per_group"] = maximum_symbols_per_group
    if margin_multiplier != 1.0:
        extra_arguments["margin_multiplier"] = margin_multiplier
        extra_arguments["margin_interest_annual_rate"] = 0.048
    evaluation_metrics = strategy.evaluate_combined_strategy(
        data_directory,
        buy_strategy_name,
        sell_strategy_name,
        minimum_average_dollar_volume=minimum_average_dollar_volume,
        top_dollar_volume_rank=top_dollar_volume_rank,
        maximum_symbols_per_group=maximum_symbols_per_group,
        minimum_average_dollar_volume_ratio=minimum_average_dollar_volume_ratio,
        starting_cash=starting_cash,
        withdraw_amount=withdraw_amount,
        stop_loss_percentage=stop_loss_percentage,
        start_date=start_date,
        allowed_fama_french_groups=allowed_groups,
        **extra_arguments,
    )
    clean_trade_details(evaluation_metrics)
    LOGGER.info(
        "%s",
        format_summary_line(strategy_identifier.upper(), evaluation_metrics),
    )
    for year_value in sorted(evaluation_metrics.annual_returns):
        annual_return = evaluation_metrics.annual_returns[year_value]
        trade_count = evaluation_metrics.annual_trade_counts.get(year_value, 0)
        LOGGER.info(
            "[%s] Year %s: %s return, %s trades",
            strategy_identifier.upper(),
            year_value,
            f"{annual_return:.2%}",
            trade_count,
        )
    if show_details:
        display_trade_details(strategy_identifier.upper(), evaluation_metrics)


def parse_arguments(argument_list: Iterable[str] | None = None) -> argparse.Namespace:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more strategy-set backtests without launching the "
            "interactive management shell."
        )
    )
    parser.add_argument(
        "--strategy-ids",
        nargs="+",
        default=["s4", "s6"],
        help=(
            "One or more strategy identifiers from data/strategy_sets.csv. "
            "Defaults to S4 and S6."
        ),
    )
    parser.add_argument(
        "--volume-filter",
        default="dollar_volume>1",
        help=(
            "Dollar-volume filter expression (same syntax as start_simulate). "
            "Defaults to 'dollar_volume>1'."
        ),
    )
    parser.add_argument(
        "--start-date",
        help=(
            "Simulation start date in YYYY-MM-DD format. "
            "Defaults to 1990-01-01."
        ),
    )
    parser.add_argument(
        "--starting-cash",
        type=float,
        default=3000.0,
        help="Initial portfolio cash. Defaults to 3000.0.",
    )
    parser.add_argument(
        "--withdraw",
        type=float,
        default=0.0,
        help="Annual withdrawal amount. Defaults to 0.0.",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=1.0,
        help="Stop-loss percentage. Defaults to 1.0.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Leverage multiplier (>=1.0). Defaults to 1.0.",
    )
    parser.add_argument(
        "--group",
        help=(
            "Comma-separated list of allowed Fama–French group identifiers "
            "(1-11). Optional."
        ),
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print individual trade details.",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_PATH),
        help=(
            "File path for writing log output. Defaults to logs/strategy_backtest.log."
        ),
    )
    return parser.parse_args([] if argument_list is None else list(argument_list))


def main(argument_list: Iterable[str] | None = None) -> None:
    """Entry point for command-line execution."""
    arguments = parse_arguments(argument_list)
    log_file_path = Path(arguments.log_file).expanduser()
    configure_logging(log_file_path)
    strategy_identifiers = parse_strategy_identifiers(arguments.strategy_ids)
    if not strategy_identifiers:
        raise ValueError("At least one strategy id is required.")
    if arguments.margin < 1.0:
        raise ValueError("Margin multiplier must be >= 1.0.")
    try:
        allowed_groups = parse_group_identifiers(arguments.group)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    price_data_directory = (
        manage.STOCK_DATA_DIRECTORY
        if manage.STOCK_DATA_DIRECTORY.exists()
        else manage.DATA_DIRECTORY
    )
    LOGGER.info("Loading price data from %s", price_data_directory)
    earliest_cached_timestamp = pandas.Timestamp(
        determine_start_date(price_data_directory)
    )
    if arguments.start_date is None:
        start_timestamp = DEFAULT_START_DATE
    else:
        try:
            start_timestamp = pandas.Timestamp(arguments.start_date)
        except ValueError as error:
            raise SystemExit(
                f"Invalid start date '{arguments.start_date}': {error}"
            ) from error
    if start_timestamp < earliest_cached_timestamp:
        LOGGER.warning(
            "Requested start date %s predates available cache %s; "
            "simulation will begin once data is available.",
            start_timestamp.date(),
            earliest_cached_timestamp.date(),
        )
    LOGGER.info("Earliest cached date: %s", earliest_cached_timestamp.date())
    LOGGER.info("Simulation start date: %s", start_timestamp.date())
    LOGGER.info("Volume filter: %s", arguments.volume_filter)
    for strategy_identifier in strategy_identifiers:
        run_strategy_set(
            strategy_identifier,
            data_directory=price_data_directory,
            volume_filter=arguments.volume_filter,
            starting_cash=arguments.starting_cash,
            withdraw_amount=arguments.withdraw,
            stop_loss_percentage=arguments.stop_loss,
            start_date=start_timestamp,
            margin_multiplier=arguments.margin,
            allowed_groups=allowed_groups,
            show_details=arguments.show_details,
        )


if __name__ == "__main__":
    main()
