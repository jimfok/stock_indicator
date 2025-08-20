"""Interactive shell for managing symbol cache and historical data."""

# TODO: review

from __future__ import annotations

import cmd
import logging
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
        """update_symbols\n        Download the latest list of ticker symbols."""
        symbols.update_symbol_cache()
        self.stdout.write("Symbol cache updated\n")

    def do_update_data(self, argument_line: str) -> None:  # noqa: D401
        """update_data SYMBOL START END\n        Download data for SYMBOL between START and END and store as CSV."""
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

    def do_update_all_data(self, argument_line: str) -> None:  # noqa: D401
        """update_all_data START END\n        Download data for all cached symbols."""
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
    def do_start_simulate(self, argument_line: str) -> None:  # noqa: D401
        """start_simulate BUY_STRATEGY SELL_STRATEGY\n        Evaluate trading strategies using cached data."""
        argument_parts: List[str] = argument_line.split()
        if len(argument_parts) != 2:
            self.stdout.write(
                "usage: start_simulate BUY_STRATEGY SELL_STRATEGY\n"
            )
            return
        buy_strategy_name, sell_strategy_name = argument_parts
        if (
            buy_strategy_name != "ema_sma_cross"
            or sell_strategy_name != "ema_sma_cross"
        ):
            self.stdout.write("unsupported strategies\n")
            return
        trade_count, win_rate = strategy.evaluate_ema_sma_cross_strategy(
            DATA_DIRECTORY
        )
        self.stdout.write(
            f"Trades: {trade_count}, Win rate: {win_rate:.2%}\n"
        )

    def do_exit(self, argument_line: str) -> bool:  # noqa: D401
        """exit\n        Exit the shell."""
        self.stdout.write("Bye\n")
        return True


if __name__ == "__main__":
    StockShell().cmdloop()
