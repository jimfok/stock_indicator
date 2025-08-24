"""Interactive shell for managing symbol cache and historical data."""

# TODO: review

from __future__ import annotations

import cmd
import logging
import re
from pathlib import Path
from typing import List

from pandas import DataFrame

from . import data_loader, symbols, strategy
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
        """start_simulate DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]
        Evaluate trading strategies using cached data.

        STOP_LOSS defaults to 1.0 when not provided."""
        argument_parts: List[str] = argument_line.split()
        if len(argument_parts) not in (3, 4):
            self.stdout.write(
                "usage: start_simulate DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]\n"
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
        volume_match = re.fullmatch(r"dollar_volume>(\d+(?:\.\d+)?)", volume_filter)
        if volume_match is None:
            self.stdout.write("unsupported filter\n")
            return
        minimum_average_dollar_volume = float(volume_match.group(1))
        if buy_strategy_name not in strategy.BUY_STRATEGIES:
            self.stdout.write("unsupported strategies\n")
            return
        if sell_strategy_name not in strategy.SELL_STRATEGIES:
            self.stdout.write("unsupported strategies\n")
            return
        evaluation_metrics = strategy.evaluate_combined_strategy(
            DATA_DIRECTORY,
            buy_strategy_name,
            sell_strategy_name,
            minimum_average_dollar_volume,
            stop_loss_percentage=stop_loss_percentage,
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

    # TODO: review
    def help_start_simulate(self) -> None:
        """Display help for the start_simulate command."""
        available_buy = ", ".join(sorted(strategy.BUY_STRATEGIES.keys()))
        available_sell = ", ".join(sorted(strategy.SELL_STRATEGIES.keys()))
        self.stdout.write(
            "start_simulate DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]\n"
            "Evaluate trading strategies using cached data.\n"
            "Parameters:\n"
            "  DOLLAR_VOLUME_FILTER: Format dollar_volume>NUMBER (in millions).\n"
            "  BUY_STRATEGY: Name of the buying strategy.\n"
            "  SELL_STRATEGY: Name of the selling strategy.\n"
            "  STOP_LOSS: Fractional loss that triggers an exit on the next day's open. "
            "Defaults to 1.0.\n"
            f"Available buy strategies: {available_buy}.\n"
            f"Available sell strategies: {available_sell}.\n"
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
