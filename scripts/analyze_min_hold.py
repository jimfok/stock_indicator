#!/usr/bin/env python3
"""
Analyze existing backtest to estimate effect of minimum holding period.

This script analyzes the existing backtest CSV to show:
1. Distribution of holding periods
2. What trades would be affected by a minimum holding period
3. Estimate of performance impact

Usage:
    python analyze_min_hold.py <backtest_csv> <min_hold_days>

Example:
    python analyze_min_hold.py logs/simulate_result/simulation_20260421_194922.csv 5
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_holding_period(entry_date: str, exit_date: str) -> int:
    """Calculate holding period in trading days."""
    entry = pd.to_datetime(entry_date)
    exit = pd.to_datetime(exit_date)
    return (exit - entry).days


def analyze_holding_periods(trades_df: pd.DataFrame, min_hold_days: int):
    """Analyze holding periods and estimate impact of min hold constraint."""
    print("\n" + "="*80)
    print(f"HOLDING PERIOD ANALYSIS - Minimum Holding: {min_hold_days} days")
    print("="*80)

    # Calculate holding periods
    trades_df['holding_days'] = trades_df.apply(
        lambda row: calculate_holding_period(row['entry_date'], row['exit_date']),
        axis=1
    )

    # Overall statistics
    total_trades = len(trades_df)
    mean_hold = trades_df['holding_days'].mean()
    median_hold = trades_df['holding_days'].median()
    min_hold = trades_df['holding_days'].min()
    max_hold = trades_df['holding_days'].max()

    print(f"\nOverall Statistics:")
    print(f"  Total trades:        {total_trades}")
    print(f"  Mean holding period: {mean_hold:.2f} days")
    print(f"  Median hold period:  {median_hold:.2f} days")
    print(f"  Min holding period:  {min_hold} days")
    print(f"  Max holding period:  {max_hold} days")

    # Distribution of holding periods
    print(f"\nHolding Period Distribution:")

    bins = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, float('inf')]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-20', '20-50', '50-100', '100+']

    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = len(trades_df[(trades_df['holding_days'] >= low) & (trades_df['holding_days'] < high)])
        pct = (count / total_trades * 100) if total_trades > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {labels[i]:6s} days: {count:4d} trades ({pct:5.1f}%) {bar}")

    # Trades affected by minimum holding period
    trades_below_min = trades_df[trades_df['holding_days'] < min_hold_days]
    trades_above_min = trades_df[trades_df['holding_days'] >= min_hold_days]

    affected_count = len(trades_below_min)
    affected_pct = (affected_count / total_trades * 100) if total_trades > 0 else 0

    print(f"\nImpact of {min_hold_days}-day minimum holding period:")
    print(f"  Trades below {min_hold_days} days:    {affected_count} ({affected_pct:.1f}%)")
    print(f"  Trades above {min_hold_days} days:    {len(trades_above_min)} ({100-affected_pct:.1f}%)")

    # Performance of short trades
    if len(trades_below_min) > 0:
        short_trades_wins = len(trades_below_min[trades_below_min['result'] == 'win'])
        short_trades_win_rate = (short_trades_wins / len(trades_below_min) * 100)
        short_trades_mean_return = trades_below_min['percentage_change'].mean() * 100

        print(f"\n  Performance of trades < {min_hold_days} days:")
        print(f"    Win rate:      {short_trades_win_rate:.2f}%")
        print(f"    Mean return:    {short_trades_mean_return:+.4f}%")

    # Performance of long trades
    if len(trades_above_min) > 0:
        long_trades_wins = len(trades_above_min[trades_above_min['result'] == 'win'])
        long_trades_win_rate = (long_trades_wins / len(trades_above_min) * 100)
        long_trades_mean_return = trades_above_min['percentage_change'].mean() * 100

        print(f"\n  Performance of trades >= {min_hold_days} days:")
        print(f"    Win rate:      {long_trades_win_rate:.2f}%")
        print(f"    Mean return:    {long_trades_mean_return:+.4f}%")

        # Compare
        if len(trades_below_min) > 0:
            win_rate_diff = long_trades_win_rate - short_trades_win_rate
            return_diff = long_trades_mean_return - short_trades_mean_return

            print(f"\n  Difference (>= {min_hold_days} days vs < {min_hold_days} days):")
            print(f"    Win rate:      {win_rate_diff:+.2f}%")
            print(f"    Mean return:    {return_diff:+.4f}%")

    # Win rate by holding period
    print(f"\nWin Rate by Holding Period:")

    hold_ranges = [(0, 5), (5, 10), (10, 20), (20, float('inf'))]
    hold_labels = ['0-5 days', '5-10 days', '10-20 days', '20+ days']

    for i, (low, high) in enumerate(hold_ranges):
        range_trades = trades_df[(trades_df['holding_days'] >= low) & (trades_df['holding_days'] < high)]
        if len(range_trades) > 0:
            wins = len(range_trades[range_trades['result'] == 'win'])
            win_rate = (wins / len(range_trades) * 100)
            mean_return = range_trades['percentage_change'].mean() * 100
            print(f"  {hold_labels[i]:12s}: {win_rate:5.1f}% ({len(range_trades):3d} trades, mean: {mean_return:+.3f}%)")

    print("\n" + "="*80)

    # Recommendation
    if len(trades_above_min) > 0 and len(trades_below_min) > 0:
        long_trades_wins = len(trades_above_min[trades_above_min['result'] == 'win'])
        long_trades_win_rate = (long_trades_wins / len(trades_above_min) * 100)
        long_trades_mean_return = trades_above_min['percentage_change'].mean() * 100

        short_trades_wins = len(trades_below_min[trades_below_min['result'] == 'win'])
        short_trades_win_rate = (short_trades_wins / len(trades_below_min) * 100)
        short_trades_mean_return = trades_below_min['percentage_change'].mean() * 100

        if long_trades_win_rate > short_trades_win_rate + 5:
            print("\n✓ RECOMMENDATION: Adding minimum holding period would IMPROVE performance")
            print(f"  Trades held >= {min_hold_days} days have {long_trades_win_rate - short_trades_win_rate:.2f}% higher win rate")
            print(f"  Consider setting minimum holding period to {min_hold_days} days")
        elif short_trades_win_rate > long_trades_win_rate + 5:
            print("\n✗ RECOMMENDATION: Minimum holding period would REDUCE performance")
            print(f"  Short trades (< {min_hold_days} days) have higher win rate")
            print(f"  Consider not adding minimum holding period")
        else:
            print(f"\n⚠ RECOMMENDATION: Minimum holding period has minimal impact")
            print(f"  Win rate difference is small: {abs(long_trades_win_rate - short_trades_win_rate):.2f}%")

        print("\n" + "="*80)


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python analyze_min_hold.py <backtest_csv> <min_hold_days>")
        print("\nExample:")
        print("  python analyze_min_hold.py logs/simulate_result/simulation_20260421_194922.csv 5")
        print("\nParameters:")
        print("  backtest_csv  - Path to backtest CSV file")
        print("  min_hold_days - Minimum holding period to analyze")
        print("\nThis script analyzes existing backtest results to show:")
        print("  - Distribution of holding periods")
        print("  - Which trades would be affected by minimum holding period")
        print("  - Performance comparison between short and long trades")
        sys.exit(1)

    backtest_csv = sys.argv[1]
    min_hold_days = int(sys.argv[2])

    try:
        # Load backtest
        trades_df = pd.read_csv(backtest_csv)

        # Analyze
        analyze_holding_periods(trades_df, min_hold_days)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
