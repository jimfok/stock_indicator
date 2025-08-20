"""Command line interface for running stock simulations.

The interface now supports selecting the price column via the
``--price-column`` option, which defaults to ``adj_close``.
"""
# TODO: review

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

import pandas

from . import data_loader, indicators, simulator, symbols

LOGGER = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the command line interface.

    The parser includes an optional ``--price-column`` argument that chooses
    which column from the price data is used for indicator calculations and
    trading rules. If not provided, ``adj_close`` is used.
    """
    parser = argparse.ArgumentParser(
        description="Run indicator calculations and trade simulations."
    )
    parser.add_argument("--symbol", required=True, help="Stock ticker symbol to use.")
    parser.add_argument(
        "--start", required=True, help="Start date for the historical data (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end", required=True, help="End date for the historical data (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["sma"],
        help="Trading strategy to apply to the data.",
    )
    parser.add_argument(
        "--price-column",
        default="adj_close",
        help=(
            "Column in the price data to use for indicator calculations and "
            "trading rules. Defaults to 'adj_close'."
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional path to a CSV file for writing the trade results.",
    )
    return parser


def run_cli(argument_list: Optional[List[str]] = None) -> None:
    """Parse command line arguments and execute the selected strategy.

    Respects the ``--price-column`` option for choosing the data column used in
    calculations.
    """
    parser = create_parser()
    parsed_arguments = parser.parse_args(argument_list)

    try:
        available_symbol_list = symbols.load_symbols()
        if (
            available_symbol_list
            and parsed_arguments.symbol not in available_symbol_list
        ):
            raise ValueError(f"Unknown symbol: {parsed_arguments.symbol}")
    except Exception as symbol_error:  # noqa: BLE001
        LOGGER.warning("Could not verify symbol: %s", symbol_error)

    price_data_frame = (
        data_loader.download_history(
            parsed_arguments.symbol, parsed_arguments.start, parsed_arguments.end
        )
        .rename(columns=str.lower)
        .rename(columns=lambda column_name: column_name.replace(" ", "_"))
    )

    price_column = parsed_arguments.price_column

    if parsed_arguments.strategy == "sma":
        indicator_series = indicators.sma(
            price_data_frame[price_column], window_size=20
        )
        price_data_frame["indicator"] = indicator_series

        def entry_rule(current_row: pandas.Series) -> bool:
            """Determine whether to open a trade for the given row."""
            row_label = current_row.name
            indicator_at_label = indicator_series.loc[row_label]
            return current_row[price_column] > indicator_at_label

        def exit_rule(
            current_row: pandas.Series, entry_row: pandas.Series
        ) -> bool:
            """Determine whether to close the current trade."""
            row_label = current_row.name
            indicator_at_label = indicator_series.loc[row_label]
            return current_row[price_column] < indicator_at_label

    else:
        raise ValueError(f"Unsupported strategy: {parsed_arguments.strategy}")

    simulation_result = simulator.simulate_trades(
        price_data_frame, entry_rule, exit_rule
    )
    LOGGER.info("Total profit: %s", simulation_result.total_profit)
    if parsed_arguments.output:
        trade_record_list = [
            {
                "entry_date": trade.entry_date,
                "exit_date": trade.exit_date,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "profit": trade.profit,
            }
            for trade in simulation_result.trades
        ]
        result_data_frame = pandas.DataFrame(trade_record_list)
        result_data_frame.to_csv(parsed_arguments.output, index=False)
        LOGGER.info("Results written to %s", parsed_arguments.output)


if __name__ == "__main__":
    run_cli()
