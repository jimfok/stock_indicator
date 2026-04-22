#!/usr/bin/env python3
"""
Calculate backtest metrics for a strategy using historical data (efficient version).

This script directly calculates entry/exit signals for symbols and simulates trades.

Strategy execution:
- Entry signal on date T → Buy at open on T+1
- Exit signal on date T → Sell at open on T+1

Usage:
    python backtest_metrics_fast.py <strategy_id> <top_n> [start_date] [end_date]

Example:
    python backtest_metrics_fast.py buy3 50 2010-01-01 2023-12-31
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings

# Suppress pandas future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    entry_signal_date: str
    entry_date: str
    entry_price: float
    exit_signal_date: str
    exit_date: str
    exit_price: float
    percentage_change: float
    result: str  # 'win' or 'lose'


@dataclass
class BacktestMetrics:
    """Summary statistics for a backtest."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    mean_win_percentage: float
    mean_loss_percentage: float
    trades_by_year: Dict[int, int]


def parse_strategy_name(strategy_name: str) -> Tuple[str, int, float, float, float, float, float, float]:
    """
    Parse strategy name to extract parameters.

    Example: "ema_sma_cross_testing_3_-99_99_-99.0,99.0_0.973,1.0"
    Returns: (base_name, window, sma_angle_min, sma_angle_max, near_min, near_max, above_min, above_max)
    """
    parts = strategy_name.split('_')

    # Default values
    base_name = parts[0]  # e.g., "ema_sma_cross_testing"
    window = 3  # Default for ema_sma_cross_testing
    sma_angle_min = -99.0
    sma_angle_max = 99.0
    near_min = 0.0
    near_max = 1.0
    above_min = 0.0
    above_max = 1.0

    # Parse parameters - look for numeric parts after underscores
    param_index = 0
    for i, part in enumerate(parts[1:], start=1):
        # Skip non-numeric parts like "cross", "testing"
        # Also handle ranges like "0.973,1.0"
        try:
            # Check if it's a range
            if ',' in part:
                range_parts = part.split(',')
                # Parse both parts
                val1 = float(range_parts[0])
                val2 = float(range_parts[1])

                param_index += 1
                if param_index == 4:
                    near_min = val1
                    near_max = val2
                    param_index += 1  # Account for second value
                elif param_index == 6:
                    above_min = val1
                    above_max = val2
                    param_index += 1
            else:
                # Single value
                val = float(part)
                param_index += 1

                if param_index == 1:
                    window = int(val)
                elif param_index == 2:
                    sma_angle_min = val
                elif param_index == 3:
                    sma_angle_max = val
                elif param_index == 4:
                    near_min = val
                elif param_index == 5:
                    near_max = val
        except ValueError:
            # Not a number, skip
            continue

    return (base_name, window, sma_angle_min, sma_angle_max, near_min, near_max, above_min, above_max)


def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=window).mean()


def calculate_angle(series: pd.Series, window: int) -> pd.Series:
    """Calculate angle of moving average."""
    ma = calculate_sma(series, window)
    # Calculate angle using arctan of relative change
    relative_change = ma.diff() / ma.shift(1)
    angle = np.degrees(np.arctan(relative_change))
    return angle


def generate_entry_signals(
    df: pd.DataFrame,
    strategy_name: str,
    dollar_volume_50d: float
) -> pd.Series:
    """
    Generate entry signals for a strategy.

    Returns boolean Series indicating entry signals (True = entry signal)
    """
    if not strategy_name.startswith('ema_sma_cross_testing'):
        # For non-testing strategies, return empty
        return pd.Series([False] * len(df), index=df.index)

    # Parse strategy parameters
    base_name, window, sma_angle_min, sma_angle_max, near_min, near_max, above_min, above_max = parse_strategy_name(strategy_name)

    # Calculate indicators
    df['ema'] = calculate_ema(df['close'], window)
    df['sma'] = calculate_sma(df['close'], window)
    df['ema_angle'] = calculate_angle(df['close'], window)
    df['sma_angle'] = calculate_angle(df['close'], window)

    # Volume profile calculations
    df['volume_price'] = df['close'] * df['volume']
    df['avg_volume_50'] = df['volume'].rolling(50).mean()
    df['volume_profile'] = df['volume_price'] / df['volume'].sum()

    # Volume concentration
    df['near_volume'] = 0.0
    df['above_volume'] = 0.0
    df['below_volume'] = 0.0

    for i in range(50, len(df)):
        current_price = df.iloc[i]['close']
        total_volume = df.iloc[i-50:i]['volume'].sum()

        near_price_vol = df.iloc[i-50:i][
            (df.iloc[i-50:i]['close'] >= current_price * 0.99) &
            (df.iloc[i-50:i]['close'] <= current_price * 1.01)
        ]['volume'].sum()

        above_price_vol = df.iloc[i-50:i][
            df.iloc[i-50:i]['close'] > current_price * 1.01
        ]['volume'].sum()

        below_price_vol = df.iloc[i-50:i][
            df.iloc[i-50:i]['close'] < current_price * 0.99
        ]['volume'].sum()

        df.loc[df.index[i], 'near_volume'] = near_price_vol / total_volume if total_volume > 0 else 0
        df.loc[df.index[i], 'above_volume'] = above_price_vol / total_volume if total_volume > 0 else 0
        df.loc[df.index[i], 'below_volume'] = below_price_vol / total_volume if total_volume > 0 else 0

    # Dollar volume filter (50-day average)
    df['dollar_volume_50d'] = (df['close'] * df['volume']).rolling(50).mean()

    # Entry signal logic
    # EMA crosses above SMA
    df['ema_above_sma'] = df['ema'] > df['sma']
    df['ema_above_sma_prev'] = df['ema_above_sma'].shift(1)
    df['cross_up'] = (df['ema_above_sma'].astype(bool) & (~df['ema_above_sma_prev'].astype(bool)))

    # Apply filters
    entry_signal = (
        df['cross_up'].shift(1) &  # Signal available for next day
        (df['sma_angle'] >= sma_angle_min) &
        (df['sma_angle'] <= sma_angle_max) &
        (df['near_volume'] >= near_min) &
        (df['near_volume'] <= near_max) &
        (df['above_volume'] >= above_min) &
        (df['above_volume'] <= above_max)
    )

    return entry_signal.fillna(False)


def generate_exit_signals(
    df: pd.DataFrame,
    strategy_name: str
) -> pd.Series:
    """
    Generate exit signals for a strategy.

    Returns boolean Series indicating exit signals (True = exit signal)
    """
    if not strategy_name.startswith('ema_sma_cross_testing'):
        return pd.Series([False] * len(df), index=df.index)

    # Parse strategy parameters
    base_name, window, sma_angle_min, sma_angle_max, near_min, near_max, above_min, above_max = parse_strategy_name(strategy_name)

    # Calculate indicators
    df['ema'] = calculate_ema(df['close'], window)
    df['sma'] = calculate_sma(df['close'], window)

    # Exit signal logic
    # EMA crosses below SMA
    df['ema_below_sma'] = df['ema'] < df['sma']
    df['ema_below_sma_prev'] = df['ema_below_sma'].shift(1)
    df['cross_down'] = (df['ema_below_sma'].astype(bool) & (~df['ema_below_sma_prev'].astype(bool)))

    exit_signal = df['cross_down'].shift(1)  # Signal available for next day
    return exit_signal.fillna(False)


def run_backtest(
    strategy_id: str,
    top_n: int = 50,
    start_date: str = None,
    end_date: str = None,
    max_symbols: int = None
) -> Tuple[List[Trade], BacktestMetrics]:
    """
    Run backtest simulation for a strategy.

    Args:
        strategy_id: Strategy ID from strategy_sets.csv (e.g., 'buy3')
        top_n: Number of top symbols by dollar volume to consider
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        max_symbols: Maximum number of symbols to process (for testing)

    Returns:
        Tuple of (list of trades, metrics)
    """
    # Load strategy sets
    strategy_file = Path(__file__).parent.parent / "data" / "strategy_sets.csv"
    strategy_df = pd.read_csv(strategy_file)

    strategy_row = strategy_df[strategy_df['strategy_id'] == strategy_id]
    if strategy_row.empty:
        logger.error(f"Strategy '{strategy_id}' not found")
        raise ValueError(f"Strategy '{strategy_id}' not found")

    buy_strategy = strategy_row.iloc[0]['buy']
    sell_strategy = strategy_row.iloc[0]['sell']

    logger.info(f"Strategy: {strategy_id}")
    logger.info(f"  Buy:  {buy_strategy}")
    logger.info(f"  Sell: {sell_strategy}")
    logger.info(f"  Top N: {top_n}")

    # Load historical data
    data_dir = Path(__file__).parent.parent / "data" / "stock_data"
    symbol_files = list(data_dir.glob("*.csv"))

    if max_symbols:
        symbol_files = symbol_files[:max_symbols]
        logger.info(f"Processing {len(symbol_files)} symbols (limited for testing)")
    else:
        logger.info(f"Processing {len(symbol_files)} symbols")

    # Load all data and calculate dollar volume
    all_data = {}
    dollar_volumes = {}

    for csv_file in symbol_files:
        symbol = csv_file.stem
        if symbol == "^GSPC":
            continue

        try:
            df = pd.read_csv(csv_file, parse_dates=['Date'])
            df.set_index('Date', inplace=True)

            # Filter by date range
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]

            if len(df) < 50:  # Need at least 50 days for calculations
                continue

            # Calculate 50-day average dollar volume
            df['dollar_volume'] = df['close'] * df['volume']
            dollar_volume_50d = df['dollar_volume'].rolling(50).mean().iloc[-1]

            all_data[symbol] = df
            dollar_volumes[symbol] = dollar_volume_50d
        except Exception as e:
            logger.debug(f"Error loading {symbol}: {e}")
            continue

    logger.info(f"Loaded {len(all_data)} symbols with valid data")

    # Select top N symbols by dollar volume
    sorted_symbols = sorted(dollar_volumes.items(), key=lambda x: x[1], reverse=True)
    top_symbols = [s[0] for s in sorted_symbols[:top_n]]

    logger.info(f"Selected top {len(top_symbols)} symbols by 50-day dollar volume")

    # Generate signals for top symbols
    trades: List[Trade] = []
    open_positions: Dict[str, Tuple[str, float]] = {}

    for symbol in top_symbols:
        if symbol not in all_data:
            continue

        df = all_data[symbol].copy()

        # Generate signals
        entry_signals = generate_entry_signals(df, buy_strategy, dollar_volumes.get(symbol, 0))
        exit_signals = generate_exit_signals(df, sell_strategy)

        # Simulate trades
        for date in df.index[1:]:  # Start from day 1 to have previous values
            entry_signal = entry_signals.get(date, False)
            exit_signal = exit_signals.get(date, False)

            # Process exits first
            if symbol in open_positions and exit_signal:
                entry_date, entry_price = open_positions[symbol]
                exit_price = df.loc[date, 'open']

                percentage_change = (exit_price - entry_price) / entry_price * 100
                result = 'win' if percentage_change > 0 else 'lose'

                trade = Trade(
                    symbol=symbol,
                    entry_signal_date=entry_date,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_signal_date=date.strftime('%Y-%m-%d'),
                    exit_date=date.strftime('%Y-%m-%d'),
                    exit_price=exit_price,
                    percentage_change=percentage_change,
                    result=result
                )
                trades.append(trade)
                del open_positions[symbol]

            # Process entries
            if entry_signal and symbol not in open_positions:
                entry_price = df.loc[date, 'open']
                open_positions[symbol] = (date.strftime('%Y-%m-%d'), entry_price)

        # Close any remaining position
        if symbol in open_positions:
            entry_date, entry_price = open_positions[symbol]
            last_date = df.index[-1]
            exit_price = df.loc[last_date, 'open']

            percentage_change = (exit_price - entry_price) / entry_price * 100
            result = 'win' if percentage_change > 0 else 'lose'

            trade = Trade(
                symbol=symbol,
                entry_signal_date=entry_date,
                entry_date=entry_date,
                entry_price=entry_price,
                exit_signal_date=last_date.strftime('%Y-%m-%d'),
                exit_date=last_date.strftime('%Y-%m-%d'),
                exit_price=exit_price,
                percentage_change=percentage_change,
                result=result
            )
            trades.append(trade)

    # Calculate metrics
    metrics = calculate_metrics(trades)
    logger.info(f"Backtest complete: {len(trades)} trades generated")

    return trades, metrics


def calculate_metrics(trades: List[Trade]) -> BacktestMetrics:
    """Calculate metrics from completed trades."""
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.result == 'win'])
    losing_trades = total_trades - winning_trades

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    winning_pct_changes = [t.percentage_change for t in trades if t.result == 'win']
    losing_pct_changes = [t.percentage_change for t in trades if t.result == 'lose']

    mean_win_percentage = sum(winning_pct_changes) / len(winning_pct_changes) if winning_pct_changes else 0
    mean_loss_percentage = sum(losing_pct_changes) / len(losing_pct_changes) if losing_pct_changes else 0

    trades_by_year = {}
    for trade in trades:
        year = pd.to_datetime(trade.entry_date).year
        trades_by_year[year] = trades_by_year.get(year, 0) + 1

    return BacktestMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        mean_win_percentage=mean_win_percentage,
        mean_loss_percentage=mean_loss_percentage,
        trades_by_year=trades_by_year
    )


def print_report(metrics: BacktestMetrics, strategy_id: str, top_n: int):
    """Print backtest metrics report."""
    print("\n" + "="*80)
    print(f"BACKTEST METRICS REPORT - {strategy_id} (Top {top_n})")
    print("="*80)

    print(f"\nOverall Statistics:")
    print(f"  Total trades:       {metrics.total_trades}")
    print(f"  Winning trades:     {metrics.winning_trades}")
    print(f"  Losing trades:      {metrics.losing_trades}")
    print(f"  Win rate:          {metrics.win_rate:.2f}%")

    print(f"\nProfit/Loss:")
    print(f"  Mean win %:        {metrics.mean_win_percentage:+.4f}%")
    print(f"  Mean loss %:       {metrics.mean_loss_percentage:+.4f}%")

    print(f"\nTrades Per Year:")
    for year in sorted(metrics.trades_by_year.keys()):
        count = metrics.trades_by_year[year]
        print(f"  {year}:              {count}")

    print("\n" + "="*80)


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python backtest_metrics_fast.py <strategy_id> <top_n> [start_date] [end_date]")
        print("\nExample:")
        print("  python backtest_metrics_fast.py buy3 50 2010-01-01 2023-12-31")
        print("\nParameters:")
        print("  strategy_id - Strategy ID from strategy_sets.csv (e.g., buy3, s3)")
        print("  top_n      - Number of top symbols by dollar volume to trade")
        print("  start_date - Optional start date (YYYY-MM-DD)")
        print("  end_date   - Optional end date (YYYY-MM-DD)")
        sys.exit(1)

    strategy_id = sys.argv[1]
    top_n = int(sys.argv[2])
    start_date = sys.argv[3] if len(sys.argv) > 3 else None
    end_date = sys.argv[4] if len(sys.argv) > 4 else None

    try:
        trades, metrics = run_backtest(strategy_id, top_n, start_date, end_date)
        print_report(metrics, strategy_id, top_n)

        # Save trades to CSV
        if trades:
            output_file = Path(__file__).parent.parent / "logs" / f"backtest_{strategy_id}_top{top_n}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            trades_df = pd.DataFrame([
                {
                    'symbol': t.symbol,
                    'entry_signal_date': t.entry_signal_date,
                    'entry_date': t.entry_date,
                    'entry_price': t.entry_price,
                    'exit_signal_date': t.exit_signal_date,
                    'exit_date': t.exit_date,
                    'exit_price': t.exit_price,
                    'percentage_change': t.percentage_change,
                    'result': t.result
                }
                for t in trades
            ])
            trades_df.to_csv(output_file, index=False)
            logger.info(f"Trades saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
