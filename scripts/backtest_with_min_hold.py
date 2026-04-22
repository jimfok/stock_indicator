#!/usr/bin/env python3
"""
Refined backtest with minimum holding period.

Rules:
1. Entry signal on date T → Buy at OPEN on T+1
2. Must hold for at least 5 days after buy execution
3. Skip exit signals in first 5 days
4. First exit signal after day 5 → Sell at OPEN on T+1

Usage:
    python backtest_with_min_hold.py <strategy_id> <top_n> <min_hold_days> [start_date] [end_date]

Example:
    python backtest_with_min_hold.py buy3 50 5 2010-01-01 2023-12-31
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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
    holding_days: int


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
    mean_holding_period: float


class RefinedBacktestSimulator:
    """Simulates trades with minimum holding period constraint."""

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

    def get_next_trading_day_open(self, symbol: str, signal_date: str) -> Optional[Tuple[str, float]]:
        """Get next trading day and its open price."""
        df = self.load_historical_data(symbol)
        if df is None:
            return None

        signal_date_parsed = pd.to_datetime(signal_date)

        # Look forward up to 7 days to find next trading day
        for offset in range(1, 8):
            candidate_date = signal_date_parsed + timedelta(days=offset)
            if candidate_date in df.index:
                open_price = df.loc[candidate_date, 'open']
                if pd.notna(open_price):
                    return candidate_date.strftime('%Y-%m-%d'), float(open_price)

        return None

    def run_backtest(
        self,
        strategy_id: str,
        top_n: int = 50,
        min_hold_days: int = 5,
        start_date: str = None,
        end_date: str = None
    ) -> Tuple[List[Trade], BacktestMetrics]:
        """Run backtest simulation with minimum holding period."""
        logger.info(f"Running refined backtest for strategy: {strategy_id}")
        logger.info(f"  Min holding period: {min_hold_days} days")

        # Load strategy sets
        strategy_file = Path(__file__).parent.parent / "data" / "strategy_sets.csv"
        strategy_df = pd.read_csv(strategy_file)

        strategy_row = strategy_df[strategy_df['strategy_id'] == strategy_id]
        if strategy_row.empty:
            logger.error(f"Strategy '{strategy_id}' not found")
            raise ValueError(f"Strategy '{strategy_id}' not found")

        buy_strategy = strategy_row.iloc[0]['buy']
        sell_strategy = strategy_row.iloc[0]['sell']

        logger.info(f"  Buy strategy:  {buy_strategy}")
        logger.info(f"  Sell strategy: {sell_strategy}")

        # Determine date range
        if start_date is None:
            start_date = '2010-01-01'
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"  Date range: {start_date} to {end_date}")

        # Simulate trades day by day
        trades: List[Trade] = []
        open_positions: Dict[str, Tuple[str, float, datetime]] = {}  # symbol -> (entry_date, entry_price, entry_datetime)

        current_date = pd.to_datetime(start_date)
        end_date_parsed = pd.to_datetime(end_date)

        day_count = 0
        total_days = (end_date_parsed - current_date).days

        logger.info(f"Simulating trades...")
        logger.info(f"  Expected {total_days} days to process")

        # Sample every month for signal checking (speed optimization)
        check_dates = pd.date_range(current_date, end_date_parsed, freq='MS')

        # Track signals between checks
        pending_entries: Dict[str, str] = {}  # symbol -> signal_date
        pending_exits: Dict[str, str] = {}  # symbol -> signal_date

        while current_date <= end_date_parsed:
            day_count += 1
            if day_count % 100 == 0:
                logger.info(f"  Processing {current_date.strftime('%Y-%m-%d')} ({day_count}/{total_days})")

            current_date_str = current_date.strftime('%Y-%m-%d')

            # Check for signals on sample dates
            if current_date in check_dates:
                try:
                    signals = find_history_signal(
                        date_string=current_date_str,
                        dollar_volume_filter="dollar_volume>0",
                        buy_strategy=buy_strategy,
                        sell_strategy=sell_strategy,
                        stop_loss=1.0
                    )

                    entry_signals = set(signals.get('entry_signals', []))
                    exit_signals = set(signals.get('exit_signals', []))

                    # Store pending signals
                    for symbol in entry_signals:
                        if symbol not in open_positions:
                            pending_entries[symbol] = current_date_str

                    for symbol in exit_signals:
                        pending_exits[symbol] = current_date_str

                except Exception as e:
                    logger.debug(f"Error getting signals for {current_date_str}: {e}")

            # Process pending entries (buy at next day's open)
            for symbol in list(pending_entries.keys()):
                if symbol not in open_positions:
                    entry_trade_date, entry_price = self.get_next_trading_day_open(symbol, current_date_str)
                    if entry_trade_date and entry_price:
                        entry_datetime = pd.to_datetime(entry_trade_date)
                        open_positions[symbol] = (entry_trade_date, entry_price, entry_datetime)
                        del pending_entries[symbol]

            # Process pending exits (with minimum holding period constraint)
            for symbol in list(open_positions.keys()):
                entry_date, entry_price, entry_datetime = open_positions[symbol]

                # Calculate holding days
                holding_days = (current_date - entry_datetime).days

                # Check if exit signal exists and holding period met
                if symbol in pending_exits and holding_days >= min_hold_days:
                    exit_signal_date = pending_exits[symbol]

                    # Execute exit at next day's open
                    exit_trade_date, exit_price = self.get_next_trading_day_open(symbol, exit_signal_date)

                    if exit_trade_date and exit_price:
                        percentage_change = (exit_price - entry_price) / entry_price * 100
                        result = 'win' if percentage_change > 0 else 'lose'

                        trade = Trade(
                            symbol=symbol,
                            entry_signal_date=exit_signal_date,  # We don't track original entry signal date
                            entry_date=entry_date,
                            entry_price=entry_price,
                            exit_signal_date=exit_signal_date,
                            exit_date=exit_trade_date,
                            exit_price=exit_price,
                            percentage_change=percentage_change,
                            result=result,
                            holding_days=holding_days
                        )
                        trades.append(trade)
                        del open_positions[symbol]
                        del pending_exits[symbol]

                # Remove expired exit signals (old signals)
                if symbol in pending_exits:
                    exit_signal_datetime = pd.to_datetime(pending_exits[symbol])
                    if (current_date - exit_signal_datetime).days > 7:
                        # Exit signal too old, remove it
                        del pending_exits[symbol]

            # Move to next day
            current_date += timedelta(days=1)

        # Close any remaining open positions
        if open_positions:
            logger.warning(f"Closing {len(open_positions)} remaining positions at end date")
            for symbol, (entry_date, entry_price, entry_datetime) in open_positions.items():
                df = self.load_historical_data(symbol)
                if df is not None:
                    last_valid_idx = df.index[df.index <= end_date_parsed]
                    if not last_valid_idx.empty:
                        last_date = last_valid_idx[-1]
                        last_price = df.loc[last_date, 'open']

                        holding_days = (last_date - entry_datetime).days
                        percentage_change = (last_price - entry_price) / entry_price * 100
                        result = 'win' if percentage_change > 0 else 'lose'

                        trade = Trade(
                            symbol=symbol,
                            entry_signal_date=entry_date,
                            entry_date=entry_date,
                            entry_price=entry_price,
                            exit_signal_date=end_date,
                            exit_date=last_date.strftime('%Y-%m-%d'),
                            exit_price=last_price,
                            percentage_change=percentage_change,
                            result=result,
                            holding_days=holding_days
                        )
                        trades.append(trade)

        # Calculate metrics
        metrics = self._calculate_metrics(trades)
        logger.info(f"Backtest complete: {len(trades)} trades generated")

        return trades, metrics

    def _calculate_metrics(self, trades: List[Trade]) -> BacktestMetrics:
        """Calculate metrics from completed trades."""
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.result == 'win'])
        losing_trades = total_trades - winning_trades

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Calculate win/loss percentages
        winning_pct_changes = [t.percentage_change for t in trades if t.result == 'win']
        losing_pct_changes = [t.percentage_change for t in trades if t.result == 'lose']

        mean_win_percentage = sum(winning_pct_changes) / len(winning_pct_changes) if winning_pct_changes else 0
        mean_loss_percentage = sum(losing_pct_changes) / len(losing_pct_changes) if losing_pct_changes else 0

        # Calculate mean holding period
        holding_days = [t.holding_days for t in trades]
        mean_holding_period = sum(holding_days) / len(holding_days) if holding_days else 0

        # Group trades by year
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
            trades_by_year=trades_by_year,
            mean_holding_period=mean_holding_period
        )

    def print_report(self, metrics: BacktestMetrics, strategy_id: str, top_n: int, min_hold_days: int):
        """Print backtest metrics report."""
        print("\n" + "="*80)
        print(f"REFINED BACKTEST METRICS - {strategy_id} (Top {top_n}, Min Hold: {min_hold_days} days)")
        print("="*80)

        print(f"\nOverall Statistics:")
        print(f"  Total trades:       {metrics.total_trades}")
        print(f"  Winning trades:     {metrics.winning_trades}")
        print(f"  Losing trades:      {metrics.losing_trades}")
        print(f"  Win rate:          {metrics.win_rate:.2f}%")

        print(f"\nProfit/Loss:")
        print(f"  Mean win %:        {metrics.mean_win_percentage:+.4f}%")
        print(f"  Mean loss %:       {metrics.mean_loss_percentage:+.4f}%")

        print(f"\nHolding Period:")
        print(f"  Mean holding days: {metrics.mean_holding_period:.2f}")

        print(f"\nTrades Per Year:")
        for year in sorted(metrics.trades_by_year.keys()):
            count = metrics.trades_by_year[year]
            print(f"  {year}:              {count}")

        print("\n" + "="*80)


def main():
    """Main entry point."""
    if len(sys.argv) < 4:
        print("Usage: python backtest_with_min_hold.py <strategy_id> <top_n> <min_hold_days> [start_date] [end_date]")
        print("\nExample:")
        print("  python backtest_with_min_hold.py buy3 50 5 2010-01-01 2023-12-31")
        print("\nParameters:")
        print("  strategy_id  - Strategy ID from strategy_sets.csv (e.g., buy3, s3)")
        print("  top_n        - Number of top symbols by dollar volume to trade")
        print("  min_hold_days - Minimum holding period in days (exit signals before this are ignored)")
        print("  start_date   - Optional start date (YYYY-MM-DD)")
        print("  end_date     - Optional end date (YYYY-MM-DD)")
        print("\nRules:")
        print("  1. Entry signal on T → Buy at OPEN on T+1")
        print("  2. Must hold for at least min_hold_days after buy")
        print("  3. Skip exit signals in first min_hold_days")
        print("  4. First exit signal after min_hold_days → Sell at OPEN on T+1")
        sys.exit(1)

    strategy_id = sys.argv[1]
    top_n = int(sys.argv[2])
    min_hold_days = int(sys.argv[3])
    start_date = sys.argv[4] if len(sys.argv) > 4 else None
    end_date = sys.argv[5] if len(sys.argv) > 5 else None

    # Initialize
    data_dir = Path(__file__).parent.parent / "data" / "stock_data"
    simulator = RefinedBacktestSimulator(str(data_dir))

    try:
        trades, metrics = simulator.run_backtest(
            strategy_id=strategy_id,
            top_n=top_n,
            min_hold_days=min_hold_days,
            start_date=start_date,
            end_date=end_date
        )

        # Print report
        simulator.print_report(metrics, strategy_id, top_n, min_hold_days)

        # Save trades to CSV
        if trades:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = Path(__file__).parent.parent / "logs" / f"refined_backtest_{strategy_id}_top{top_n}_min{min_hold_days}_{timestamp}.csv"
            trades_df = pd.DataFrame([
                {
                    'symbol': t.symbol,
                    'entry_date': t.entry_date,
                    'entry_price': t.entry_price,
                    'exit_signal_date': t.exit_signal_date,
                    'exit_date': t.exit_date,
                    'exit_price': t.exit_price,
                    'percentage_change': t.percentage_change,
                    'result': t.result,
                    'holding_days': t.holding_days
                }
                for t in trades
            ])
            trades_df.to_csv(output_file, index=False)
            logger.info(f"Trades saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
