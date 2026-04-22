#!/usr/bin/env python3
"""
Run refined backtest with minimum holding period constraint.

This script uses existing backtest results and applies minimum holding period
by checking historical data for exit signals.

Rules:
1. Entry on original entry_date, entry_price
2. Must hold for at least min_hold_days
3. Find first exit signal after min_hold_days
4. Exit at next day's open

Usage:
    python run_refined_backtest.py <original_csv> <min_hold_days>

Example:
    python run_refined_backtest.py logs/simulate_result/simulation_20260421_194922.csv 5
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
class RefinedTrade:
    """Trade with minimum holding period applied."""
    symbol: str
    entry_date: str
    entry_price: float
    exit_signal_date: str
    exit_date: str
    exit_price: float
    percentage_change: float
    result: str
    holding_days: int
    original_exit_date: str
    original_exit_price: float
    original_pct_change: float


class RefinedBacktester:
    """Applies minimum holding period to existing backtest."""

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

    def find_exit_after_min_hold(
        self,
        symbol: str,
        entry_date: str,
        min_hold_days: int,
        original_exit_date: str,
        sell_strategy: str
    ) -> Tuple[Optional[str], Optional[float], int]:
        """
        Find exit signal after minimum holding period.

        Returns:
            (exit_signal_date, exit_price, holding_days)
        """
        df = self.load_historical_data(symbol)
        if df is None:
            return None, None, 0

        entry_datetime = pd.to_datetime(entry_date)
        original_exit_datetime = pd.to_datetime(original_exit_date)

        # Start checking from entry_date + min_hold_days
        check_start = entry_datetime + timedelta(days=min_hold_days)

        # Check every day for exit signals (up to original exit date)
        check_date = check_start
        while check_date <= original_exit_datetime:
            if check_date not in df.index:
                check_date += timedelta(days=1)
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
                    # Found exit signal - get price at next open
                    next_day = check_date + timedelta(days=1)
                    max_attempts = 10

                    for attempt in range(max_attempts):
                        if next_day in df.index:
                            exit_price = df.loc[next_day, 'open']
                            if pd.notna(exit_price):
                                holding_days = (next_day - entry_datetime).days
                                return check_date.strftime('%Y-%m-%d'), float(exit_price), holding_days
                        next_day += timedelta(days=1)

            except Exception as e:
                logger.debug(f"Error checking {symbol} at {check_date}: {e}")

            check_date += timedelta(days=1)

        # No exit signal found - use original exit date/price
        original_exit_price = self.get_price_at_date(symbol, original_exit_date, 'open')
        holding_days = (original_exit_datetime - entry_datetime).days

        return original_exit_date, original_exit_price, holding_days

    def run_refined_backtest(
        self,
        original_csv: str,
        min_hold_days: int = 5,
        sell_strategy: str = "ema_sma_cross_testing_3_-0.01_65_-10.0,10.0_0.78,1.00"
    ) -> List[RefinedTrade]:
        """Run refined backtest with minimum holding period."""
        logger.info(f"Loading original backtest: {original_csv}")

        # Load original trades
        original_df = pd.read_csv(original_csv)

        logger.info(f"Loaded {len(original_df)} trades")
        logger.info(f"Applying {min_hold_days}-day minimum holding period")

        refined_trades: List[RefinedTrade] = []

        for i, row in original_df.iterrows():
            if (i + 1) % 50 == 0:
                logger.info(f"Processing {i + 1}/{len(original_df)} trades")

            symbol = row['symbol']
            entry_date = row['entry_date']
            original_exit_date = row['exit_date']
            original_pct_change = row['percentage_change'] * 100

            # Get entry price
            entry_price = self.get_price_at_date(symbol, entry_date, 'open')
            if entry_price is None:
                logger.warning(f"Could not find entry price for {symbol} on {entry_date}")
                continue

            # Get original exit price
            original_exit_price = self.get_price_at_date(symbol, original_exit_date, 'open')

            # Find refined exit after minimum holding period
            refined_exit_date, refined_exit_price, holding_days = self.find_exit_after_min_hold(
                symbol,
                entry_date,
                min_hold_days,
                original_exit_date,
                sell_strategy
            )

            if refined_exit_date and refined_exit_price:
                # Calculate refined return
                refined_pct_change = (refined_exit_price - entry_price) / entry_price * 100
                result = 'win' if refined_pct_change > 0 else 'lose'

                refined_trade = RefinedTrade(
                    symbol=symbol,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_signal_date=refined_exit_date,
                    exit_date=refined_exit_date,
                    exit_price=refined_exit_price,
                    percentage_change=refined_pct_change,
                    result=result,
                    holding_days=holding_days,
                    original_exit_date=original_exit_date,
                    original_exit_price=original_exit_price or 0.0,
                    original_pct_change=original_pct_change
                )
                refined_trades.append(refined_trade)

        logger.info(f"Generated {len(refined_trades)} refined trades")

        return refined_trades

    def print_comparison(self, original_df: pd.DataFrame, refined_trades: List[RefinedTrade]):
        """Print comparison between original and refined backtest."""
        print("\n" + "="*80)
        print("REFAINED BACKTEST COMPARISON")
        print("="*80)

        # Original metrics
        original_total = len(original_df)
        original_wins = len(original_df[original_df['result'] == 'win'])
        original_win_rate = (original_wins / original_total * 100) if original_total > 0 else 0
        original_mean_pct = original_df['percentage_change'].mean() * 100

        # Refined metrics
        refined_total = len(refined_trades)
        refined_wins = len([t for t in refined_trades if t.result == 'win'])
        refined_win_rate = (refined_wins / refined_total * 100) if refined_total > 0 else 0
        refined_mean_pct = sum([t.percentage_change for t in refined_trades]) / refined_total if refined_total > 0 else 0

        # Calculate CAGR (simplified)
        original_cagr = original_mean_pct * 100 / 14  # Approximate over 14 years
        refined_cagr = refined_mean_pct * 100 / 14

        print(f"\nOriginal Backtest (buy3):")
        print(f"  Total trades:   {original_total}")
        print(f"  Win rate:       {original_win_rate:.2f}%")
        print(f"  Mean return:    {original_mean_pct:+.4f}%")
        print(f"  Est. CAGR:      {original_cagr:+.2f}%")

        print(f"\nRefined Backtest ({len(refined_trades[0].holding_days) if refined_trades else 5}-day min hold):")
        print(f"  Total trades:   {refined_total}")
        print(f"  Win rate:       {refined_win_rate:.2f}%")
        print(f"  Mean return:    {refined_mean_pct:+.4f}%")
        print(f"  Est. CAGR:      {refined_cagr:+.2f}%")

        # Improvement
        win_rate_improvement = refined_win_rate - original_win_rate
        return_improvement = refined_mean_pct - original_mean_pct
        cagr_improvement = refined_cagr - original_cagr

        print(f"\nImprovement:")
        print(f"  Win rate:       {win_rate_improvement:+.2f}%")
        print(f"  Mean return:    {return_improvement:+.4f}%")
        print(f"  Est. CAGR:      {cagr_improvement:+.2f}%")

        # Check profitability
        if refined_mean_pct > 0:
            print(f"\n✓ REFINED BACKTEST IS PROFITABLE")
        else:
            print(f"\n✗ REFINED BACKTEST STILL LOSING")

        # Holding period stats
        holding_days = [t.holding_days for t in refined_trades]
        mean_hold = sum(holding_days) / len(holding_days) if holding_days else 0

        print(f"\nHolding Period (refined):")
        print(f"  Mean:           {mean_hold:.2f} days")
        print(f"  Min:            {min(holding_days) if holding_days else 0} days")
        print(f"  Max:            {max(holding_days) if holding_days else 0} days")

        print("\n" + "="*80)

        return {
            'original_total': original_total,
            'original_win_rate': original_win_rate,
            'original_mean_pct': original_mean_pct,
            'refined_total': refined_total,
            'refined_win_rate': refined_win_rate,
            'refined_mean_pct': refined_mean_pct,
            'win_rate_improvement': win_rate_improvement,
            'return_improvement': return_improvement
        }


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python run_refined_backtest.py <original_csv> <min_hold_days>")
        print("\nExample:")
        print("  python run_refined_backtest.py logs/simulate_result/simulation_20260421_194922.csv 5")
        print("\nParameters:")
        print("  original_csv   - Path to original backtest CSV")
        print("  min_hold_days  - Minimum holding period in days")
        print("\nThis script applies minimum holding period to existing backtest:")
        print("  1. Loads original trades")
        print("  2. For each trade, finds first exit signal after min_hold_days")
        print("  3. Calculates new return at that exit")
        print("  4. Compares with original results")
        sys.exit(1)

    original_csv = sys.argv[1]
    min_hold_days = int(sys.argv[2])

    # Initialize
    data_dir = Path(__file__).parent.parent / "data" / "stock_data"
    backtester = RefinedBacktester(str(data_dir))

    try:
        # Load original for comparison
        original_df = pd.read_csv(original_csv)

        # Run refined backtest
        refined_trades = backtester.run_refined_backtest(original_csv, min_hold_days)

        # Print comparison
        results = backtester.print_comparison(original_df, refined_trades)

        # Save refined trades
        if refined_trades:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = Path(__file__).parent.parent / "logs" / f"refined_backtest_buy3_min{min_hold_days}_{timestamp}.csv"
            trades_df = pd.DataFrame([
                {
                    'symbol': t.symbol,
                    'entry_date': t.entry_date,
                    'entry_price': t.entry_price,
                    'exit_signal_date': t.exit_signal_date,
                    'exit_date': t.exit_date,
                    'exit_price': t.exit_price,
                    'percentage_change': t.percentage_change / 100,  # Back to decimal
                    'result': t.result,
                    'holding_days': t.holding_days,
                    'original_exit_date': t.original_exit_date,
                    'original_pct_change': t.original_pct_change / 100  # Back to decimal
                }
                for t in refined_trades
            ])
            trades_df.to_csv(output_file, index=False)
            logger.info(f"Refined trades saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
