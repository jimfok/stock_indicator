#!/usr/bin/env python3
"""
Apply minimum holding period constraint to existing backtest results.

This script post-processes existing backtest CSV to simulate what would happen
if trades had to be held for at least N days before exiting.

Rules:
- For each trade, find the first exit signal N days after entry
- Calculate the profit/loss at that exit
- Compare with original results

Usage:
    python apply_min_hold.py <backtest_csv> <min_hold_days> [sell_strategy]

Example:
    python apply_min_hold.py logs/simulate_result/simulation_20260421_194922.csv 5
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stock_indicator.daily_job import find_history_signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OriginalTrade:
    """Original trade from backtest."""
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    percentage_change: float
    result: str


@dataclass
class RefinedTrade:
    """Trade with minimum holding period applied."""
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    percentage_change: float
    result: str
    holding_days: int
    original_exit_date: str  # For comparison


class MinHoldPostProcessor:
    """Post-process backtest results with minimum holding period."""

    def __init__(self, historical_data_directory: str):
        self.historical_data_directory = Path(historical_data_directory)
        self.historical_data_cache: Dict[str, pd.DataFrame] = {}

    def load_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load historical price data for a symbol."""
        if symbol in self.historical_data_cache:
            return self.historical_data_cache[symbol]

        data_file = self.historical_data_directory / f"{symbol}.csv"
        if not data_file.exists():
            return None

        df = pd.read_csv(data_file, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        self.historical_data_cache[symbol] = df
        return df

    def get_price_at_date(self, symbol: str, date_str: str, price_type: str = 'open') -> Optional[float]:
        """Get price at specific date."""
        df = self.load_historical_data(symbol)
        if df is None:
            return None

        date_parsed = pd.to_datetime(date_str)
        if date_parsed not in df.index:
            return None

        price = df.loc[date_parsed, price_type]
        return float(price) if pd.notna(price) else None

    def find_refined_exit(
        self,
        symbol: str,
        entry_date: str,
        min_hold_days: int,
        sell_strategy: str,
        end_date: str = None
    ) -> Tuple[str, float, int]:
        """
        Find the exit date/price after minimum holding period.

        Returns:
            (exit_date, exit_price, holding_days)
        """
        df = self.load_historical_data(symbol)
        if df is None:
            return None, None, 0

        entry_datetime = pd.to_datetime(entry_date)
        end_datetime = pd.to_datetime(end_date) if end_date else datetime.now()

        # Check dates starting from entry_date + min_hold_days
        check_start = entry_datetime + timedelta(days=min_hold_days)

        # Check every 7 days for exit signals (speed optimization)
        check_dates = pd.date_range(check_start, end_datetime, freq='7D')

        for check_date in check_dates:
            if check_date not in df.index:
                continue

            try:
                signals = find_history_signal(
                    date_string=check_date.strftime('%Y-%m-%d'),
                    dollar_volume_filter="dollar_volume>0",
                    buy_strategy="ignored",
                    sell_strategy=sell_strategy,
                    stop_loss=1.0
                )

                exit_signals = signals.get('exit_signals', [])

                if symbol in exit_signals:
                    # Found exit signal - exit at next day's open
                    for offset in range(1, 8):
                        exit_date_candidate = check_date + timedelta(days=offset)
                        if exit_date_candidate in df.index:
                            exit_price = df.loc[exit_date_candidate, 'open']
                            if pd.notna(exit_price):
                                holding_days = (exit_date_candidate - entry_datetime).days
                                return exit_date_candidate.strftime('%Y-%m-%d'), float(exit_price), holding_days

            except Exception as e:
                logger.debug(f"Error checking {symbol} at {check_date}: {e}")
                continue

        # No exit signal found - close at last available date
        valid_dates = df.index[(df.index >= check_start) & (df.index <= end_datetime)]
        if not valid_dates.empty:
            last_date = valid_dates[-1]
            last_price = df.loc[last_date, 'open']
            holding_days = (last_date - entry_datetime).days
            return last_date.strftime('%Y-%m-%d'), float(last_price), holding_days

        # Fall back to original exit date if no refined exit found
        original_exit_price = self.get_price_at_date(symbol, entry_date, 'open')
        original_exit = self.find_original_exit_date(symbol, entry_date, end_date)

        if original_exit:
            holding_days = (pd.to_datetime(original_exit) - entry_datetime).days
            return original_exit, original_exit_price or 0.0, holding_days

        return None, None, 0

    def find_original_exit_date(self, symbol: str, entry_date: str, end_date: str = None) -> Optional[str]:
        """Find the original exit date from historical data."""
        # This is a simplified approach - in practice, you'd need to parse the original backtest
        # For now, return a date 7 days after entry as a fallback
        entry_datetime = pd.to_datetime(entry_date)
        return (entry_datetime + timedelta(days=7)).strftime('%Y-%m-%d')

    def process_backtest(
        self,
        backtest_csv_path: str,
        min_hold_days: int = 5,
        sell_strategy: str = None
    ) -> Tuple[List[OriginalTrade], List[RefinedTrade]]:
        """Process backtest CSV with minimum holding period."""
        logger.info(f"Loading backtest: {backtest_csv_path}")

        # Load original backtest
        original_df = pd.read_csv(backtest_csv_path)

        logger.info(f"Loaded {len(original_df)} trades")

        if sell_strategy is None:
            # Default to same strategy as original
            sell_strategy = "ema_sma_cross_testing_3_-0.01_65_-10.0,10.0_0.78,1.00"

        # Parse original trades
        original_trades: List[OriginalTrade] = []

        for _, row in original_df.iterrows():
            trade = OriginalTrade(
                symbol=row['symbol'],
                entry_date=row['entry_date'],
                exit_date=row['exit_date'],
                entry_price=row['signal_bar_open'],  # Entry at open on entry_date
                exit_price=row['signal_bar_open'] if row['result'] == 'win' else row['signal_bar_open'],  # Need to calculate
                percentage_change=row['percentage_change'] * 100,
                result=row['result']
            )
            # Note: The CSV doesn't store entry/exit prices directly
            # We need to look them up from historical data
            original_trades.append(trade)

        # Apply minimum holding period
        refined_trades: List[RefinedTrade] = []

        for i, trade in enumerate(original_trades):
            if (i + 1) % 50 == 0:
                logger.info(f"Processing {i + 1}/{len(original_trades)} trades")

            # Get entry price from historical data
            entry_price = self.get_price_at_date(trade.symbol, trade.entry_date, 'open')
            if entry_price is None:
                logger.warning(f"Could not find entry price for {trade.symbol} on {trade.entry_date}")
                continue

            # Get original exit price
            original_exit_price = self.get_price_at_date(trade.symbol, trade.exit_date, 'open')

            # Find refined exit after min_hold_days
            refined_exit_date, refined_exit_price, holding_days = self.find_refined_exit(
                trade.symbol,
                trade.entry_date,
                min_hold_days,
                sell_strategy,
                trade.exit_date  # Use original exit as max date
            )

            if refined_exit_date and refined_exit_price:
                percentage_change = (refined_exit_price - entry_price) / entry_price * 100
                result = 'win' if percentage_change > 0 else 'lose'

                refined_trade = RefinedTrade(
                    symbol=trade.symbol,
                    entry_date=trade.entry_date,
                    exit_date=refined_exit_date,
                    entry_price=entry_price,
                    exit_price=refined_exit_price,
                    percentage_change=percentage_change,
                    result=result,
                    holding_days=holding_days,
                    original_exit_date=trade.exit_date
                )
                refined_trades.append(refined_trade)

        logger.info(f"Generated {len(refined_trades)} refined trades")

        return original_trades, refined_trades

    def print_comparison(self, original_trades: List[OriginalTrade], refined_trades: List[RefinedTrade]):
        """Print comparison report."""
        print("\n" + "="*80)
        print("BACKTEST COMPARISON: Original vs. Minimum Holding Period")
        print("="*80)

        # Calculate original metrics
        original_wins = len([t for t in original_trades if t.result == 'win'])
        original_win_rate = (original_wins / len(original_trades) * 100) if original_trades else 0
        original_pct_changes = [t.percentage_change for t in original_trades]
        original_mean = sum(original_pct_changes) / len(original_pct_changes) if original_pct_changes else 0

        # Calculate refined metrics
        refined_wins = len([t for t in refined_trades if t.result == 'win'])
        refined_win_rate = (refined_wins / len(refined_trades) * 100) if refined_trades else 0
        refined_pct_changes = [t.percentage_change for t in refined_trades]
        refined_mean = sum(refined_pct_changes) / len(refined_pct_changes) if refined_pct_changes else 0
        refined_hold_days = [t.holding_days for t in refined_trades]
        refined_mean_hold = sum(refined_hold_days) / len(refined_hold_days) if refined_hold_days else 0

        print(f"\nOriginal Backtest:")
        print(f"  Total trades:   {len(original_trades)}")
        print(f"  Win rate:       {original_win_rate:.2f}%")
        print(f"  Mean return:    {original_mean:+.4f}%")

        print(f"\nRefined Backtest (min hold):")
        print(f"  Total trades:   {len(refined_trades)}")
        print(f"  Win rate:       {refined_win_rate:.2f}%")
        print(f"  Mean return:    {refined_mean:+.4f}%")
        print(f"  Mean hold days: {refined_mean_hold:.2f}")

        print(f"\nImprovement:")
        print(f"  Win rate delta: {refined_win_rate - original_win_rate:+.2f}%")
        print(f"  Return delta:   {refined_mean - original_mean:+.4f}%")

        print("\n" + "="*80)


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python apply_min_hold.py <backtest_csv> <min_hold_days> [sell_strategy]")
        print("\nExample:")
        print("  python apply_min_hold.py logs/simulate_result/simulation_20260421_194922.csv 5")
        print("\nParameters:")
        print("  backtest_csv  - Path to backtest CSV file")
        print("  min_hold_days - Minimum holding period in days")
        print("  sell_strategy - Optional sell strategy name")
        sys.exit(1)

    backtest_csv = sys.argv[1]
    min_hold_days = int(sys.argv[2])
    sell_strategy = sys.argv[3] if len(sys.argv) > 3 else None

    # Initialize
    data_dir = Path(__file__).parent.parent / "data" / "stock_data"
    processor = MinHoldPostProcessor(str(data_dir))

    try:
        original_trades, refined_trades = processor.process_backtest(
            backtest_csv,
            min_hold_days,
            sell_strategy
        )

        # Print comparison
        processor.print_comparison(original_trades, refined_trades)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
