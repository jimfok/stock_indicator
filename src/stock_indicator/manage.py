"""Interactive shell for managing symbol cache and historical data."""

# TODO: review

from __future__ import annotations

import cmd
import datetime
import logging
import re
from pathlib import Path
from typing import Dict, List

from pandas import DataFrame

from . import data_loader, symbols, strategy, volume, daily_job
from .daily_job import determine_start_date
from .symbols import SP500_SYMBOL

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

    def do_update_data(self, argument_line: str) -> None:  # noqa: D401
        """update_data SYMBOL START END
        Download data for SYMBOL between START and END and store as CSV."""
        argument_parts: List[str] = argument_line.split()
        if len(argument_parts) != 3:
            self.stdout.write("usage: update_data SYMBOL START END\n")
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

    # TODO: review
    def help_update_data(self) -> None:
        """Display help for the update_data command."""
        self.stdout.write(
            "update_data SYMBOL START END\n"
            "Download data for SYMBOL between START and END and store as CSV.\n"
            "Parameters:\n"
            "  SYMBOL: Ticker symbol for the asset.\n"
            "  START: Start date in YYYY-MM-DD format.\n"
            "  END: End date in YYYY-MM-DD format.\n"
        )

    def do_update_all_data(self, argument_line: str) -> None:  # noqa: D401
        """update_all_data START END
        Download data for all cached symbols."""
        argument_parts: List[str] = argument_line.split()
        if len(argument_parts) != 2:
            self.stdout.write("usage: update_all_data START END\n")
            return
        start_date, end_date = argument_parts
        symbol_list = symbols.load_symbols()
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

    # TODO: review
    def help_update_all_data(self) -> None:
        """Display help for the update_all_data command."""
        self.stdout.write(
            "update_all_data START END\n"
            "Download data for all cached symbols.\n"
            "Parameters:\n"
            "  START: Start date in YYYY-MM-DD format.\n"
            "  END: End date in YYYY-MM-DD format.\n"
        )

    # TODO: review
    def do_start_simulate(self, argument_line: str) -> None:  # noqa: D401
        """start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]
        Evaluate trading strategies using cached data.

        STOP_LOSS defaults to 1.0 when not provided."""
        argument_parts: List[str] = argument_line.split()
        starting_cash_value = 3000.0
        withdraw_amount = 0.0
        while argument_parts and (
            argument_parts[0].startswith("starting_cash=")
            or argument_parts[0].startswith("withdraw=")
        ):
            parameter_part = argument_parts.pop(0)
            name, value = parameter_part.split("=", 1)
            try:
                numeric_value = float(value)
            except ValueError:
                self.stdout.write(f"invalid {name}\n")
                return
            if name == "starting_cash":
                starting_cash_value = numeric_value
            elif name == "withdraw":
                withdraw_amount = numeric_value
        if len(argument_parts) not in (3, 4):
            self.stdout.write(
                "usage: start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]\n"
            )
            return
        volume_filter, buy_strategy_name, sell_strategy_name = argument_parts[:3]
        if len(argument_parts) == 4:
            try:
                stop_loss_percentage = float(argument_parts[3])
            except ValueError:
                self.stdout.write("invalid stop loss\n")
                return
        else:
            stop_loss_percentage = 1.0
        minimum_average_dollar_volume: float | None = None  # TODO: review
        top_dollar_volume_rank: int | None = None  # TODO: review
        combined_match = re.fullmatch(
            r"dollar_volume>(\d+(?:\.\d+)?),(\d+)th",
            volume_filter,
        )
        if combined_match is not None:
            minimum_average_dollar_volume = float(combined_match.group(1))
            top_dollar_volume_rank = int(combined_match.group(2))
        else:
            volume_match = re.fullmatch(r"dollar_volume>(\d+(?:\.\d+)?)", volume_filter)
            if volume_match is not None:
                minimum_average_dollar_volume = float(volume_match.group(1))
            else:
                rank_match = re.fullmatch(r"dollar_volume=(\d+)th", volume_filter)
                if rank_match is not None:
                    top_dollar_volume_rank = int(rank_match.group(1))
                else:
                    self.stdout.write(
                        "unsupported filter; expected dollar_volume>NUMBER, "
                        "dollar_volume=RANKth, or dollar_volume>NUMBER,RANKth\n",
                    )
                    return
        if buy_strategy_name not in strategy.BUY_STRATEGIES:
            self.stdout.write("unsupported strategies\n")
            return
        if sell_strategy_name not in strategy.SELL_STRATEGIES:
            self.stdout.write("unsupported strategies\n")
            return

        start_date_string = determine_start_date(DATA_DIRECTORY)
        evaluation_metrics = strategy.evaluate_combined_strategy(
            DATA_DIRECTORY,
            buy_strategy_name,
            sell_strategy_name,
            minimum_average_dollar_volume=minimum_average_dollar_volume,
            top_dollar_volume_rank=top_dollar_volume_rank,
            starting_cash=starting_cash_value,
            withdraw_amount=withdraw_amount,
            stop_loss_percentage=stop_loss_percentage,
        )
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
                f"Final balance: {evaluation_metrics.final_balance:.2f}\n"
            )
        )
        for year, annual_return in sorted(
            evaluation_metrics.annual_returns.items()
        ):
            trade_count = evaluation_metrics.annual_trade_counts.get(year, 0)
            self.stdout.write(
                f"Year {year}: {annual_return:.2%}, trade: {trade_count}\n"
            )
            trade_details = evaluation_metrics.trade_details_by_year.get(year, [])  # TODO: review
            for trade_detail in trade_details:  # TODO: review
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
            "start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]\n"
            "Evaluate trading strategies using cached data.\n"
            "Parameters:\n"
            "  starting_cash: Initial cash balance for the simulation. Defaults to 3000.\n"
            "  withdraw: Amount deducted from cash at each year end. Defaults to 0.\n"
            "  DOLLAR_VOLUME_FILTER: Use dollar_volume>NUMBER (in millions),\n"
            "    dollar_volume=Nth to select the N symbols with the highest\n"
            "    previous-day dollar volume, or dollar_volume>NUMBER,Nth to\n"
            "    apply both filters.\n"
            "  BUY_STRATEGY: Name of the buying strategy.\n"
            "  SELL_STRATEGY: Name of the selling strategy.\n"
            "  STOP_LOSS: Fractional loss that triggers an exit on the next day's open. Defaults to 1.0.\n"
            f"Available buy strategies: {available_buy}.\n"
            f"Available sell strategies: {available_sell}.\n"
        )

# TODO: review
    def do_count_symbols_with_average_dollar_volume_above(self, argument_line: str) -> None:  # noqa: D401
        """count_symbols_with_average_dollar_volume_above THRESHOLD
        Count symbols whose 50-day average dollar volume exceeds THRESHOLD."""
        argument_string = argument_line.strip()
        if not argument_string:
            self.stdout.write(
                "usage: count_symbols_with_average_dollar_volume_above THRESHOLD\n"
            )
            return
        try:
            minimum_average_dollar_volume = float(argument_string)
        except ValueError:
            self.stdout.write(
                "usage: count_symbols_with_average_dollar_volume_above THRESHOLD\n"
            )
            return
        symbol_count = volume.count_symbols_with_average_dollar_volume_above(
            DATA_DIRECTORY, minimum_average_dollar_volume
        )
        self.stdout.write(f"{symbol_count}\n")

    # TODO: review
    def help_count_symbols_with_average_dollar_volume_above(self) -> None:
        """Display help for the count_symbols_with_average_dollar_volume_above command."""
        self.stdout.write(
            "count_symbols_with_average_dollar_volume_above THRESHOLD\n"
            "Count symbols whose 50-day average dollar volume is greater than THRESHOLD in millions.\n"
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
