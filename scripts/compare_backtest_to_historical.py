#!/usr/bin/env python3
"""
Compare backtest results against historical stock data to validate accuracy.

This script:
1. Loads backtest results from simulation CSV files
2. Loads historical price data for traded symbols
3. Validates entry/exit prices and percentage changes
4. Reports discrepancies and generates a summary

Usage:
    python scripts/compare_backtest_to_historical.py <backtest_csv_path>

Example:
    python scripts/compare_backtest_to_historical.py logs/simulate_result/simulation_20260421_104010.csv
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TradeDiscrepancy:
    """Represents a discrepancy between backtest and historical data."""
    symbol: str
    entry_date: str
    exit_date: str
    backtest_percentage: float
    calculated_percentage: float
    difference: float
    possible_causes: List[str]


@dataclass
class ValidationReport:
    """Summary of backtest validation results."""
    total_trades: int
    validated_trades: int
    trades_with_data_missing: int
    trades_with_discrepancies: int
    discrepancies: List[TradeDiscrepancy]
    mean_absolute_error: float
    max_error: float
    error_threshold: float = 0.5  # 0.5% tolerance


class BacktestValidator:
    """Validates backtest results against historical price data."""

    def __init__(self, historical_data_directory: str):
        """
        Initialize the validator.

        Args:
            historical_data_directory: Path to directory containing historical stock CSV files
        """
        self.historical_data_directory = Path(historical_data_directory)
        self.historical_data_cache: Dict[str, pd.DataFrame] = {}

    def load_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load historical price data for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with columns: Date, close, high, low, open, volume
            or None if file not found
        """
        # Check cache first
        if symbol in self.historical_data_cache:
            return self.historical_data_cache[symbol]

        # Try to load from CSV file
        data_file = self.historical_data_directory / f"{symbol}.csv"

        if not data_file.exists():
            logger.warning(f"Historical data file not found: {data_file}")
            return None

        try:
            df = pd.read_csv(data_file)
            # Ensure Date column is parsed
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            self.historical_data_cache[symbol] = df
            return df
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return None

    def calculate_percentage_change(
        self,
        symbol: str,
        entry_date: str,
        exit_date: str
    ) -> Optional[float]:
        """
        Calculate percentage change from historical data.

        Args:
            symbol: Stock ticker symbol
            entry_date: Entry date string (YYYY-MM-DD)
            exit_date: Exit date string (YYYY-MM-DD)

        Returns:
            Percentage change as float, or None if data unavailable
        """
        df = self.load_historical_data(symbol)
        if df is None:
            return None

        entry_date_parsed = pd.to_datetime(entry_date)
        exit_date_parsed = pd.to_datetime(exit_date)

        # Check if dates exist in data
        if entry_date_parsed not in df.index:
            logger.warning(f"Entry date {entry_date} not found for {symbol}")
            return None

        if exit_date_parsed not in df.index:
            logger.warning(f"Exit date {exit_date} not found for {symbol}")
            return None

        # Get open price on entry date and open price on exit date
        # Backtest enters at open on entry date, exits at open on exit date
        entry_price = df.loc[entry_date_parsed, 'open']
        exit_price = df.loc[exit_date_parsed, 'open']

        if pd.isna(entry_price) or pd.isna(exit_price):
            return None

        percentage_change = (exit_price - entry_price) / entry_price * 100
        return percentage_change

    def validate_trade(
        self,
        trade: pd.Series,
        error_threshold: float = 0.5
    ) -> Optional[TradeDiscrepancy]:
        """
        Validate a single trade against historical data.

        Args:
            trade: Pandas Series containing trade information
            error_threshold: Percentage difference threshold for flagging discrepancies

        Returns:
            TradeDiscrepancy if discrepancy found, None otherwise
        """
        symbol = trade['symbol']
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']

        # Backtest CSV stores percentage as decimal (e.g., 0.074 = 7.4%)
        # Convert to percentage format for comparison
        backtest_percentage = trade['percentage_change'] * 100

        # Calculate expected percentage from historical data
        calculated_percentage = self.calculate_percentage_change(
            symbol,
            entry_date,
            exit_date
        )

        if calculated_percentage is None:
            return None

        # Calculate difference
        difference = abs(backtest_percentage - calculated_percentage)

        # Check if difference exceeds threshold
        if difference > error_threshold:
            # Determine possible causes
            possible_causes = self._identify_discrepancy_causes(
                trade,
                backtest_percentage,
                calculated_percentage
            )

            return TradeDiscrepancy(
                symbol=symbol,
                entry_date=entry_date,
                exit_date=exit_date,
                backtest_percentage=backtest_percentage,
                calculated_percentage=calculated_percentage,
                difference=difference,
                possible_causes=possible_causes
            )

        return None

    def _identify_discrepancy_causes(
        self,
        trade: pd.Series,
        backtest_percentage: float,
        calculated_percentage: float
    ) -> List[str]:
        """
        Identify possible causes for a discrepancy.

        Args:
            trade: Trade data
            backtest_percentage: Percentage from backtest
            calculated_percentage: Percentage calculated from historical data

        Returns:
            List of possible causes
        """
        causes = []

        # Check if signs match
        if (backtest_percentage >= 0) != (calculated_percentage >= 0):
            causes.append("Sign mismatch - backtest says win, historical says lose (or vice versa)")

        # Check exit reason
        exit_reason = trade.get('exit_reason', 'unknown')
        if exit_reason == 'stop_loss':
            causes.append("Stop loss may have triggered at different price than open")
        elif exit_reason == 'end_of_data':
            causes.append("Trade forced to close at end of backtest period")

        # Check if difference is large
        if abs(backtest_percentage - calculated_percentage) > 10:
            causes.append("Large percentage difference - possible data source issue")

        # Check holding period
        entry_date = pd.to_datetime(trade['entry_date'])
        exit_date = pd.to_datetime(trade['exit_date'])
        holding_period = (exit_date - entry_date).days

        if holding_period == 0:
            causes.append("Zero-day holding period - may be same-day trade")

        return causes

    def validate_backtest(
        self,
        backtest_csv_path: str,
        error_threshold: float = 0.5
    ) -> ValidationReport:
        """
        Validate entire backtest against historical data.

        Args:
            backtest_csv_path: Path to backtest CSV file
            error_threshold: Percentage difference threshold for flagging discrepancies

        Returns:
            ValidationReport with summary statistics
        """
        logger.info(f"Loading backtest results from: {backtest_csv_path}")

        # Load backtest CSV
        try:
            backtest_df = pd.read_csv(backtest_csv_path)
            logger.info(f"Loaded {len(backtest_df)} trades from backtest")
        except Exception as e:
            logger.error(f"Error loading backtest CSV: {e}")
            raise

        # Initialize report
        report = ValidationReport(
            total_trades=len(backtest_df),
            validated_trades=0,
            trades_with_data_missing=0,
            trades_with_discrepancies=0,
            discrepancies=[],
            mean_absolute_error=0.0,
            max_error=0.0,
            error_threshold=error_threshold
        )

        errors = []

        # Validate each trade
        for index, trade in backtest_df.iterrows():
            discrepancy = self.validate_trade(trade, error_threshold)

            if discrepancy is not None:
                report.discrepancies.append(discrepancy)
                report.trades_with_discrepancies += 1
                errors.append(discrepancy.difference)
            else:
                # Check if we could even validate it
                calculated_percentage = self.calculate_percentage_change(
                    trade['symbol'],
                    trade['entry_date'],
                    trade['exit_date']
                )
                if calculated_percentage is None:
                    report.trades_with_data_missing += 1
                else:
                    report.validated_trades += 1
                    # Calculate error with backtest percentage already converted to %
                    backtest_pct = trade['percentage_change'] * 100
                    errors.append(abs(backtest_pct - calculated_percentage))

            # Progress logging
            if (index + 1) % 50 == 0:
                logger.info(f"Validated {index + 1}/{len(backtest_df)} trades")

        # Calculate error statistics
        if errors:
            report.mean_absolute_error = sum(errors) / len(errors)
            report.max_error = max(errors)

        logger.info(f"Validation complete: {report.validated_trades} validated, "
                   f"{report.trades_with_discrepancies} discrepancies")

        return report

    def print_report(self, report: ValidationReport):
        """
        Print validation report to console.

        Args:
            report: ValidationReport to print
        """
        print("\n" + "="*80)
        print("BACKTEST VALIDATION REPORT")
        print("="*80)

        print(f"\nSummary:")
        print(f"  Total trades:          {report.total_trades}")
        print(f"  Successfully validated: {report.validated_trades}")
        print(f"  Missing data:           {report.trades_with_data_missing}")
        print(f"  Discrepancies found:    {report.trades_with_discrepancies}")

        if report.validated_trades > 0:
            discrepancy_rate = (report.trades_with_discrepancies / report.validated_trades) * 100
            print(f"  Discrepancy rate:       {discrepancy_rate:.2f}%")

        print(f"\nError Statistics:")
        print(f"  Mean absolute error:   {report.mean_absolute_error:.4f}%")
        print(f"  Max error:             {report.max_error:.4f}%")
        print(f"  Error threshold:       ±{report.error_threshold:.2f}%")

        if report.discrepancies:
            print(f"\nTop 10 Discrepancies (sorted by error magnitude):")
            print("-"*80)

            # Sort discrepancies by difference
            sorted_discrepancies = sorted(
                report.discrepancies,
                key=lambda x: x.difference,
                reverse=True
            )

            for i, discrepancy in enumerate(sorted_discrepancies[:10], 1):
                print(f"\n{i}. {discrepancy.symbol}")
                print(f"   Entry: {discrepancy.entry_date} → Exit: {discrepancy.exit_date}")
                print(f"   Backtest:    {discrepancy.backtest_percentage:+.4f}%")
                print(f"   Historical:  {discrepancy.calculated_percentage:+.4f}%")
                print(f"   Difference:  {discrepancy.difference:.4f}%")
                if discrepancy.possible_causes:
                    print(f"   Possible causes:")
                    for cause in discrepancy.possible_causes:
                        print(f"     - {cause}")

        print("\n" + "="*80)


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python compare_backtest_to_historical.py <backtest_csv_path>")
        print("\nExample:")
        print("  python compare_backtest_to_historical.py logs/simulate_result/simulation_20260421_104010.csv")
        sys.exit(1)

    backtest_csv_path = sys.argv[1]

    if not Path(backtest_csv_path).exists():
        logger.error(f"Backtest CSV file not found: {backtest_csv_path}")
        sys.exit(1)

    # Determine historical data directory
    script_dir = Path(__file__).parent.parent
    historical_data_dir = script_dir / "data" / "stock_data"

    if not historical_data_dir.exists():
        logger.error(f"Historical data directory not found: {historical_data_dir}")
        sys.exit(1)

    # Initialize validator
    validator = BacktestValidator(str(historical_data_dir))

    # Validate backtest
    try:
        report = validator.validate_backtest(backtest_csv_path)

        # Print report
        validator.print_report(report)

        # Exit with error code if discrepancies found
        if report.trades_with_discrepancies > 0:
            logger.warning(f"Found {report.trades_with_discrepancies} discrepancies")
            sys.exit(1)
        else:
            logger.info("Validation passed - no discrepancies found")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
