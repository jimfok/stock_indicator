"""Interactive shell for managing symbol cache and historical data."""

# TODO: review

from __future__ import annotations

import cmd
import datetime
import logging
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import pandas
from pandas import DataFrame

from . import data_loader, symbols, strategy, daily_job
from .daily_job import determine_start_date
from .symbols import SP500_SYMBOL
from stock_indicator.sector_pipeline import pipeline

LOGGER = logging.getLogger(__name__)

DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"


class StockShell(cmd.Cmd):
    """Interactive command shell for stock data maintenance."""

    intro = "Stock Indicator shell. Type help or ? to list commands."
    prompt = "(stock-indicator) "

    def do_update_symbols(self, argument_line: str) -> None:  # noqa: D401
        """update_symbols
        Download the latest list of ticker symbols."""
        symbols.update_symbol_cache()
        self.stdout.write("Symbol cache updated\n")

    # TODO: review
    def help_update_symbols(self) -> None:
        """Display help for the update_symbols command."""
        self.stdout.write(
            "update_symbols\n"
            "Download the latest list of ticker symbols.\n"
            "This command has no parameters.\n"
        )

    def do_update_yf_symbols(self, argument_line: str) -> None:  # noqa: D401
        """update_yf_symbols
        Build a list of symbols that have data available from Yahoo Finance."""
        yf_symbols = symbols.update_yf_symbol_cache()
        self.stdout.write(f"YF symbol cache updated (count={len(yf_symbols)})\n")

    def help_update_yf_symbols(self) -> None:
        """Display help for the update_yf_symbols command."""
        self.stdout.write(
            "update_yf_symbols\n"
            "Probe Yahoo Finance and write symbols with available data to data/symbols_yf.txt.\n"
            "This command has no parameters.\n"
        )

    def do_update_data_from_yf(self, argument_line: str) -> None:  # noqa: D401
        """update_data_from_yf SYMBOL START END
        Download data from Yahoo Finance for SYMBOL between START and END and store as CSV."""
        argument_parts: List[str] = argument_line.split()
        if len(argument_parts) != 3:
            self.stdout.write("usage: update_data_from_yf SYMBOL START END\n")
            return
        symbol_name, start_date, end_date = argument_parts
        data_frame: DataFrame = data_loader.download_history(
            symbol_name, start_date, end_date
        )
        data_frame_with_date: DataFrame = (
            data_frame.reset_index().rename(columns={"index": "Date"})
        )
        DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output_path = DATA_DIRECTORY / f"{symbol_name}.csv"
        data_frame_with_date.to_csv(output_path, index=False)
        self.stdout.write(f"Data written to {output_path}\n")
        # Also ensure S&P 500 index data is maintained separately when updating a single symbol
        if symbol_name != SP500_SYMBOL:
            sp_frame: DataFrame = data_loader.download_history(
                SP500_SYMBOL, start_date, end_date
            )
            sp_with_date: DataFrame = (
                sp_frame.reset_index().rename(columns={"index": "Date"})
            )
            sp_output = DATA_DIRECTORY / f"{SP500_SYMBOL}.csv"
            sp_with_date.to_csv(sp_output, index=False)
            self.stdout.write(f"Data written to {sp_output}\n")

    def help_update_data_from_yf(self) -> None:
        """Display help for the update_data_from_yf command."""
        self.stdout.write(
            "update_data_from_yf SYMBOL START END\n"
            "Download data from Yahoo Finance for SYMBOL and write CSV to data/<SYMBOL>.csv.\n"
            "Parameters:\n"
            "  SYMBOL: Ticker symbol for the asset.\n"
            "  START: Start date in YYYY-MM-DD format.\n"
            "  END: End date in YYYY-MM-DD format.\n"
        )

    

    def do_update_all_data_from_yf(self, argument_line: str) -> None:  # noqa: D401
        """update_all_data_from_yf START END
        Download data from Yahoo Finance for all cached symbols."""
        argument_parts: List[str] = argument_line.split()
        if len(argument_parts) != 2:
            self.stdout.write("usage: update_all_data_from_yf START END\n")
            return
        start_date, end_date = argument_parts
        symbol_list = symbols.load_symbols()
        # Ensure ^GSPC is also downloaded, but not stored in symbols.txt
        if SP500_SYMBOL not in symbol_list:
            symbol_list.append(SP500_SYMBOL)
        for symbol_name in symbol_list:
            data_frame: DataFrame = data_loader.download_history(
                symbol_name, start_date, end_date
            )
            data_frame_with_date: DataFrame = (
                data_frame.reset_index().rename(columns={"index": "Date"})
            )
            DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
            output_path = DATA_DIRECTORY / f"{symbol_name}.csv"
            data_frame_with_date.to_csv(output_path, index=False)
            self.stdout.write(f"Data written to {output_path}\n")

    def help_update_all_data_from_yf(self) -> None:
        """Display help for the update_all_data_from_yf command."""
        self.stdout.write(
            "update_all_data_from_yf START END\n"
            "Download data from Yahoo Finance for all cached symbols.\n"
            "Parameters:\n"
            "  START: Start date in YYYY-MM-DD format.\n"
            "  END: End date in YYYY-MM-DD format.\n"
        )

    def do_update_sector_data(self, argument_line: str) -> None:  # noqa: D401
        """update_sector_data [--ff-map-url=URL OUTPUT_PATH]
        Refresh the local sector classification data set."""
        argument_parts: List[str] = argument_line.split()
        if not argument_parts:
            LOGGER.info(
                "Updating sector classification data using last run configuration",
            )
            try:
                data_frame: DataFrame = pipeline.update_latest_dataset()
            except (FileNotFoundError, ValueError, OSError) as error:
                self.stdout.write(
                    f"Error: {error}\n"
                    "usage: update_sector_data --ff-map-url=URL OUTPUT_PATH\n"
                )
                return
            coverage_report = pipeline.generate_coverage_report(data_frame)
            self.stdout.write(f"{coverage_report}\n")
            return
        mapping_url: str | None = None
        output_path_string: str | None = None
        for token in argument_parts:
            if token.startswith("--ff-map-url="):
                mapping_url = token.split("=", 1)[1]
            else:
                output_path_string = token
        if mapping_url is None or output_path_string is None:
            self.stdout.write(
                "usage: update_sector_data --ff-map-url=URL OUTPUT_PATH\n",
            )
            return
        output_path = Path(output_path_string)
        LOGGER.info(
            "Building sector classification data using %s",
            mapping_url,
        )
        data_frame = pipeline.build_sector_classification_dataset(
            mapping_url,
            output_path,
        )
        coverage_report = pipeline.generate_coverage_report(data_frame)
        self.stdout.write(f"{coverage_report}\n")

    def help_update_sector_data(self) -> None:
        """Display help for the update_sector_data command."""
        self.stdout.write(
            "update_sector_data --ff-map-url=URL OUTPUT_PATH\n"
            "Refresh sector classification data from SEC and Fama-French sources.\n"
            "The ticker universe is sourced from the SEC company tickers dataset.\n"
            "Without parameters, rebuilds data using the last saved configuration.\n"
            "Parameters:\n"
            "  --ff-map-url: URL or file path to SIC to Fama-French mapping.\n"
            "  OUTPUT_PATH: Destination path for the Parquet output file.\n"
        )

    # TODO: review
    def do_start_simulate(self, argument_line: str) -> None:  # noqa: D401
        """start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS] [SHOW_DETAILS]
        Evaluate trading strategies using cached data.

        STOP_LOSS defaults to 1.0 when not provided.
        SHOW_DETAILS defaults to True and controls whether trade details are printed."""
        argument_parts: List[str] = argument_line.split()
        starting_cash_value = 3000.0
        withdraw_amount = 0.0
        start_date_string: str | None = None
        while argument_parts and (
            argument_parts[0].startswith("starting_cash=")
            or argument_parts[0].startswith("withdraw=")
            or argument_parts[0].startswith("start=")
        ):
            parameter_part = argument_parts.pop(0)
            name, value = parameter_part.split("=", 1)
            if name == "start":
                try:
                    datetime.date.fromisoformat(value)
                except ValueError:
                    self.stdout.write("invalid start date\n")
                    return
                start_date_string = value
                continue
            try:
                numeric_value = float(value)
            except ValueError:
                self.stdout.write(f"invalid {name}\n")
                return
            if name == "starting_cash":
                starting_cash_value = numeric_value
            elif name == "withdraw":
                withdraw_amount = numeric_value
        if len(argument_parts) not in (3, 4, 5):
            self.stdout.write(
                "usage: start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS] [SHOW_DETAILS]\n"
            )
            return
        volume_filter, buy_strategy_name, sell_strategy_name = argument_parts[:3]
        stop_loss_percentage = 1.0
        show_trade_details = True  # TODO: review
        if len(argument_parts) >= 4:
            try:
                stop_loss_percentage = float(argument_parts[3])
            except ValueError:
                self.stdout.write("invalid stop loss\n")
                return
        if len(argument_parts) == 5:
            show_trade_details = argument_parts[4].lower() == "true"  # TODO: review
        minimum_average_dollar_volume: float | None = None  # TODO: review
        minimum_average_dollar_volume_ratio: float | None = None  # TODO: review
        top_dollar_volume_rank: int | None = None  # TODO: review
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
                            self.stdout.write(
                                "unsupported filter; expected dollar_volume>NUMBER, "
                                "dollar_volume>NUMBER%, dollar_volume=RANKth, "
                                "dollar_volume>NUMBER,RANKth, or "
                                "dollar_volume>NUMBER%,RANKth\n",
                            )
                            return
        try:  # TODO: review
            buy_base_name, _, _ = strategy.parse_strategy_name(buy_strategy_name)
            sell_base_name, _, _ = strategy.parse_strategy_name(sell_strategy_name)
        except ValueError as error:
            self.stdout.write(f"{error}\n")
            return
        if (
            buy_base_name not in strategy.BUY_STRATEGIES
            or sell_base_name not in strategy.SELL_STRATEGIES
        ):
            self.stdout.write("unsupported strategies\n")
            return

        if start_date_string is None:
            start_date_string = determine_start_date(DATA_DIRECTORY)
        start_timestamp = pandas.Timestamp(start_date_string)
        evaluation_metrics = strategy.evaluate_combined_strategy(
            DATA_DIRECTORY,
            buy_strategy_name,
            sell_strategy_name,
            minimum_average_dollar_volume=minimum_average_dollar_volume,
            top_dollar_volume_rank=top_dollar_volume_rank,
            minimum_average_dollar_volume_ratio=minimum_average_dollar_volume_ratio,
            starting_cash=starting_cash_value,
            withdraw_amount=withdraw_amount,
            stop_loss_percentage=stop_loss_percentage,
            start_date=start_timestamp,
        )
        earliest_valid_googl_date = datetime.date(2014, 4, 3)
        filtered_trade_details_by_year: Dict[int, List[strategy.TradeDetail]] = {}
        removed_any_trade = False
        for year, trade_list in evaluation_metrics.trade_details_by_year.items():
            cleaned_trade_list = []
            for trade_detail in trade_list:
                if (
                    trade_detail.symbol == "GOOGL"
                    and trade_detail.date.date() < earliest_valid_googl_date
                ):
                    removed_any_trade = True
                    continue
                cleaned_trade_list.append(trade_detail)
            if cleaned_trade_list:
                filtered_trade_details_by_year[year] = cleaned_trade_list
        evaluation_metrics.trade_details_by_year = filtered_trade_details_by_year
        if removed_any_trade:
            all_trade_details = sorted(
                (
                    trade_detail
                    for year_trades in filtered_trade_details_by_year.values()
                    for trade_detail in year_trades
                ),
                key=lambda detail: detail.date,
            )
            close_trade_details = [
                trade_detail
                for trade_detail in all_trade_details
                if trade_detail.action == "close"
            ]
            winning_changes = [
                trade_detail.percentage_change
                for trade_detail in close_trade_details
                if trade_detail.result == "win"
                and trade_detail.percentage_change is not None
            ]
            losing_changes = [
                -trade_detail.percentage_change
                for trade_detail in close_trade_details
                if trade_detail.result == "lose"
                and trade_detail.percentage_change is not None
            ]
            open_positions: Dict[str, pandas.Timestamp] = {}
            holding_periods: List[int] = []
            for trade_detail in all_trade_details:
                if trade_detail.action == "open":
                    open_positions[trade_detail.symbol] = trade_detail.date
                elif trade_detail.action == "close":
                    entry_date = open_positions.pop(trade_detail.symbol, None)
                    if entry_date is not None:
                        holding_periods.append(
                            (trade_detail.date - entry_date).days
                        )
            evaluation_metrics.total_trades = len(close_trade_details)
            evaluation_metrics.win_rate = (
                len(winning_changes) / len(close_trade_details)
                if close_trade_details
                else 0.0
            )
            evaluation_metrics.mean_profit_percentage = (
                mean(winning_changes) if winning_changes else 0.0
            )
            evaluation_metrics.profit_percentage_standard_deviation = (
                stdev(winning_changes) if len(winning_changes) > 1 else 0.0
            )
            evaluation_metrics.mean_loss_percentage = (
                mean(losing_changes) if losing_changes else 0.0
            )
            evaluation_metrics.loss_percentage_standard_deviation = (
                stdev(losing_changes) if len(losing_changes) > 1 else 0.0
            )
            evaluation_metrics.mean_holding_period = (
                mean(holding_periods) if holding_periods else 0.0
            )
            evaluation_metrics.holding_period_standard_deviation = (
                stdev(holding_periods) if len(holding_periods) > 1 else 0.0
            )
            evaluation_metrics.annual_trade_counts = {
                year: sum(
                    1 for trade_detail in details if trade_detail.action == "close"
                )
                for year, details in filtered_trade_details_by_year.items()
            }
            evaluation_metrics.annual_returns = {
                year: annual_return
                for year, annual_return in evaluation_metrics.annual_returns.items()
                if year in filtered_trade_details_by_year
            }
        self.stdout.write(
            f"Simulation start date: {start_date_string}\n"
        )
        self.stdout.write(
            (
                f"Trades: {evaluation_metrics.total_trades}, "
                f"Win rate: {evaluation_metrics.win_rate:.2%}, "
                f"Mean profit %: {evaluation_metrics.mean_profit_percentage:.2%}, "
                f"Profit % Std Dev: {evaluation_metrics.profit_percentage_standard_deviation:.2%}, "
                f"Mean loss %: {evaluation_metrics.mean_loss_percentage:.2%}, "
                f"Loss % Std Dev: {evaluation_metrics.loss_percentage_standard_deviation:.2%}, "
                f"Mean holding period: {evaluation_metrics.mean_holding_period:.2f} bars, "
                f"Holding period Std Dev: {evaluation_metrics.holding_period_standard_deviation:.2f} bars, "
                f"Max concurrent positions: {evaluation_metrics.maximum_concurrent_positions}, "
                f"Final balance: {evaluation_metrics.final_balance:.2f}, "
                f"CAGR: {evaluation_metrics.compound_annual_growth_rate:.2%}, "
                f"Max drawdown: {evaluation_metrics.maximum_drawdown:.2%}\n"
            )
        )
        for year, annual_return in sorted(
            evaluation_metrics.annual_returns.items()
        ):
            trade_count = evaluation_metrics.annual_trade_counts.get(year, 0)
            self.stdout.write(
                f"Year {year}: {annual_return:.2%}, trade: {trade_count}\n"
            )
            if show_trade_details:  # TODO: review
                trade_details = evaluation_metrics.trade_details_by_year.get(year, [])
                for trade_detail in trade_details:
                    if (
                        trade_detail.action == "close"
                        and trade_detail.result is not None
                    ):
                        if trade_detail.percentage_change is not None:
                            result_suffix = (
                                f" {trade_detail.result} "
                                f"{trade_detail.percentage_change:.2%}"
                            )
                        else:
                            result_suffix = f" {trade_detail.result}"
                    else:
                        result_suffix = ""
                    self.stdout.write(
                        (
                            f"  {trade_detail.date.date()} {trade_detail.symbol} "
                            f"{trade_detail.action} {trade_detail.price:.2f} "
                            f"{trade_detail.simple_moving_average_dollar_volume_ratio:.4f} "
                            f"{trade_detail.simple_moving_average_dollar_volume / 1_000_000:.2f}M "
                            f"{trade_detail.total_simple_moving_average_dollar_volume / 1_000_000:.2f}M"
                            f"{result_suffix}\n"
                        )
                    )

    # TODO: review
    def help_start_simulate(self) -> None:
        """Display help for the start_simulate command."""
        available_buy = ", ".join(sorted(strategy.BUY_STRATEGIES.keys()))
        available_sell = ", ".join(sorted(strategy.SELL_STRATEGIES.keys()))
        self.stdout.write(
            "start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS] [SHOW_DETAILS]\n"
            "Evaluate trading strategies using cached data.\n"
            "Parameters:\n"
            "  starting_cash: Initial cash balance for the simulation. Defaults to 3000.\n"
            "  withdraw: Amount deducted from cash at each year end. Defaults to 0.\n"
            "  start: Date in YYYY-MM-DD format to begin the simulation. Defaults to the earliest available date.\n"
            "  DOLLAR_VOLUME_FILTER: Use dollar_volume>NUMBER (in millions),\n"
            "    dollar_volume>N% to require the 50-day average dollar volume to\n"
            "    exceed N percent of the total market, dollar_volume=Nth to\n"
            "    select the N symbols with the highest previous-day dollar\n"
            "    volume, or combine the threshold with ranking using\n"
            "    dollar_volume>NUMBER,Nth or dollar_volume>N%,Nth.\n"
            "  BUY_STRATEGY: Name of the buying strategy.\n"
            "  SELL_STRATEGY: Name of the selling strategy.\n"
            "  STOP_LOSS: Fractional loss that triggers an exit on the next day's open. Defaults to 1.0.\n"
            "  SHOW_DETAILS: 'True' to print individual trades, 'False' to suppress them. Defaults to True.\n"
            "Strategies may be suffixed with _N to set the window size to N; the default window size is 40 when no suffix is provided.\n"
            "Slope-aware strategies follow the ema_sma_signal_with_slope_n_k pattern and accept _LOWER_UPPER bounds after the optional window size; both bounds are floating-point numbers and may be negative.\n"
            "Example: start_simulate start=1990-01-01 dollar_volume>50 ema_sma_cross_20 ema_sma_cross_20\n"
            "Another example: start_simulate dollar_volume>1 ema_sma_signal_with_slope_-0.1_1.2 ema_sma_signal_with_slope_-0.1_1.2\n"
            f"Available buy strategies: {available_buy}.\n"
            f"Available sell strategies: {available_sell}.\n"
        )

    

    # TODO: review
    def do_find_signal(self, argument_line: str) -> None:  # noqa: D401
        """find_signal DATE DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY STOP_LOSS
        Display the entry and exit signals generated for DATE."""
        usage_message = (
            "usage: find_signal DATE DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY STOP_LOSS\n"
        )
        argument_parts: List[str] = argument_line.split()
        if len(argument_parts) != 5:
            self.stdout.write(usage_message)
            return
        (
            date_string,
            dollar_volume_filter,
            buy_strategy_name,
            sell_strategy_name,
            stop_loss_string,
        ) = argument_parts
        try:
            datetime.date.fromisoformat(date_string)
        except ValueError:
            self.stdout.write(usage_message)
            return
        try:
            stop_loss_value = float(stop_loss_string)
        except ValueError:
            self.stdout.write("invalid stop loss\n")
            return
        signal_data: Dict[str, List[str]] = daily_job.find_signal(
            date_string,
            dollar_volume_filter,
            buy_strategy_name,
            sell_strategy_name,
            stop_loss_value,
        )
        entry_signal_list: List[str] = signal_data.get("entry_signals", [])
        exit_signal_list: List[str] = signal_data.get("exit_signals", [])
        self.stdout.write(f"{entry_signal_list}\n")
        self.stdout.write(f"{exit_signal_list}\n")

    # TODO: review
    def help_find_signal(self) -> None:
        """Display help for the find_signal command."""
        self.stdout.write(
            "find_signal DATE DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY STOP_LOSS\n"
            "Display entry and exit signals for DATE using the provided strategies.\n"
        )


    def do_exit(self, argument_line: str) -> bool:  # noqa: D401
        """exit
        Exit the shell."""
        self.stdout.write("Bye\n")
        return True

    # TODO: review
    def help_exit(self) -> None:
        """Display help for the exit command."""
        self.stdout.write("exit\nExit the shell.\n")


if __name__ == "__main__":
    StockShell().cmdloop()
