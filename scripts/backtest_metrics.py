#!/usr/bin/env python3
"""
Calculate backtest metrics for a strategy using historical data.

This script simulates trading based on entry/exit signals and calculates:
- Win rate
- Mean win percentage (for winning trades)
- Trades per year

Strategy execution:
- Entry signal on date T → Buy at open on T+1
- Exit signal on date T → Sell at open on T+1

Usage:
    python backtest_metrics.py <strategy_id> [start_date] [end_date]

Example:
    python backtest_metrics.py buy3 2010-01-01 2023-12-31
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

# Define data directory directly
STOCK_DATA_DIRECTORY = Path(__file__).parent.parent / "data" / "stock_data"

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


class BacktestSimulator:
    """Simulates trades based on entry/exit signals."""

    def __init__(self, historical_data_directory: str):
        """
        Initialize simulator.

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
            DataFrame with index as Date, columns: close, high, low, open, volume
            or None if file not found
        """
        # Check cache
        if symbol in self.historical_data_cache:
            return self.historical_data_cache[symbol]

        # Load from CSV
        data_file = self.historical_data_directory / f"{symbol}.csv"
        if not data_file.exists():
            return None

        df = pd.read_csv(data_file, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        self.historical_data_cache[symbol] = df
        return df

    def get_next_trading_day_open(self, symbol: str, signal_date: str) -> Optional[Tuple[str, float]]:
        """
        Get the next trading day and its open price.

        Args:
            symbol: Stock ticker symbol
            signal_date: Date string (YYYY-MM-DD) when signal was generated

        Returns:
            Tuple of (next_trading_day, open_price) or None if unavailable
        """
        df = self.load_historical_data(symbol)
        if df is None:
            return None

        signal_date_parsed = pd.to_datetime(signal_date)

        # Find the next trading day (signal_date + 1 day)
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
        start_date: str = None,
        end_date: str = None
    ) -> Tuple[List[Trade], BacktestMetrics]:
        """
        Run backtest simulation for a strategy.

        Args:
            strategy_id: Strategy ID from strategy_sets.csv (e.g., 'buy3')
            start_date: Optional start date (YYYY-MM-DD), uses oldest available if None
            end_date: Optional end date (YYYY-MM-DD), uses latest available if None

        Returns:
            Tuple of (list of trades, metrics)
        """
        logger.info(f"Starting backtest for strategy: {strategy_id}")

        # Parse strategy_sets.csv to get buy/sell strategies
        strategy_sets = self._load_strategy_sets()
        if strategy_id not in strategy_sets:
            logger.error(f"Strategy ID '{strategy_id}' not found in strategy_sets.csv")
            raise ValueError(f"Strategy ID '{strategy_id}' not found")

        strategy_config = strategy_sets[strategy_id]
        buy_strategy = strategy_config['buy']
        sell_strategy = strategy_config['sell']

        logger.info(f"  Buy strategy:  {buy_strategy}")
        logger.info(f"  Sell strategy: {sell_strategy}")

        # Determine date range
        all_dates = self._get_all_trading_dates()
        if start_date is None:
            start_date = all_dates[0]
        if end_date is None:
            end_date = all_dates[-1]

        logger.info(f"  Date range: {start_date} to {end_date}")

        # Simulate trades day by day
        trades: List[Trade] = []
        open_positions: Dict[str, Tuple[str, float]] = {}  # symbol -> (entry_date, entry_price)

        current_date = pd.to_datetime(start_date)
        end_date_parsed = pd.to_datetime(end_date)

        day_count = 0
        total_days = (end_date_parsed - current_date).days

        while current_date <= end_date_parsed:
            day_count += 1
            if day_count % 100 == 0:
                logger.info(f"Processing {current_date.strftime('%Y-%m-%d')} ({day_count}/{total_days})")

            current_date_str = current_date.strftime('%Y-%m-%d')

            # Get entry/exit signals for this date
            try:
                signals = find_history_signal(
                    date_string=current_date_str,
                    dollar_volume_filter="dollar_volume>0",  # No volume filter
                    buy_strategy=buy_strategy,
                    sell_strategy=sell_strategy,
                    stop_loss=1.0  # No stop loss
                )
            except Exception as e:
                logger.debug(f"Error getting signals for {current_date_str}: {e}")
                current_date += timedelta(days=1)
                continue

            entry_signals = set(signals.get('entry_signals', []))
            exit_signals = set(signals.get('exit_signals', []))

            # Process exit signals first (sell at next day's open)
            for symbol in list(open_positions.keys()):
                if symbol in exit_signals:
                    entry_date, entry_price = open_positions[symbol]
                    exit_trade_date, exit_price = self.get_next_trading_day_open(symbol, current_date_str)

                    if exit_trade_date and exit_price:
                        # Calculate percentage change
                        percentage_change = (exit_price - entry_price) / entry_price * 100
                        result = 'win' if percentage_change > 0 else 'lose'

                        trade = Trade(
                            symbol=symbol,
                            entry_signal_date=entry_date,
                            entry_date=entry_date,
                            entry_price=entry_price,
                            exit_signal_date=current_date_str,
                            exit_date=exit_trade_date,
                            exit_price=exit_price,
                            percentage_change=percentage_change,
                            result=result
                        )
                        trades.append(trade)
                        del open_positions[symbol]

            # Process entry signals (buy at next day's open)
            for symbol in entry_signals:
                if symbol not in open_positions:
                    entry_trade_date, entry_price = self.get_next_trading_day_open(symbol, current_date_str)
                    if entry_trade_date and entry_price:
                        open_positions[symbol] = (entry_trade_date, entry_price)

            # Move to next day
            current_date += timedelta(days=1)

        # Close any remaining open positions at the last available price
        if open_positions:
            logger.warning(f"Closing {len(open_positions)} remaining positions at end date")
            for symbol, (entry_date, entry_price) in open_positions.items():
                df = self.load_historical_data(symbol)
                if df is not None:
                    # Get last available price (should be near end_date)
                    last_valid_idx = df.index[df.index <= end_date_parsed]
                    if not last_valid_idx.empty:
                        last_date = last_valid_idx[-1]
                        last_price = df.loc[last_date, 'open']
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
                            result=result
                        )
                        trades.append(trade)

        # Calculate metrics
        metrics = self._calculate_metrics(trades)

        logger.info(f"Backtest complete: {len(trades)} trades generated")

        return trades, metrics

    def _load_strategy_sets(self) -> Dict[str, Dict]:
        """Load strategy configurations from strategy_sets.csv."""
        strategy_file = Path(__file__).parent.parent / "data" / "strategy_sets.csv"
        df = pd.read_csv(strategy_file)

        strategies = {}
        for _, row in df.iterrows():
            strategies[row['strategy_id']] = {
                'buy': row['buy'],
                'sell': row['sell']
            }

        return strategies

    def _get_all_trading_dates(self) -> List[str]:
        """
        Get all unique trading dates across all symbols.

        Returns:
            List of date strings sorted ascending
        """
        dates = set()

        for csv_file in self.historical_data_directory.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, parse_dates=['Date'])
                dates.update(df['Date'].dt.strftime('%Y-%m-%d'))
            except Exception:
                continue

        sorted_dates = sorted(list(dates))
        return sorted_dates

    def _calculate_metrics(self, trades: List[Trade]) -> BacktestMetrics:
        """
        Calculate metrics from completed trades.

        Args:
            trades: List of completed trades

        Returns:
            BacktestMetrics object with calculated statistics
        """
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.result == 'win'])
        losing_trades = total_trades - winning_trades

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Calculate win/loss percentages
        winning_pct_changes = [t.percentage_change for t in trades if t.result == 'win']
        losing_pct_changes = [t.percentage_change for t in trades if t.result == 'lose']

        mean_win_percentage = sum(winning_pct_changes) / len(winning_pct_changes) if winning_pct_changes else 0
        mean_loss_percentage = sum(losing_pct_changes) / len(losing_pct_changes) if losing_pct_changes else 0

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
            trades_by_year=trades_by_year
        )

    def print_report(self, metrics: BacktestMetrics):
        """
        Print backtest metrics report.

        Args:
            metrics: BacktestMetrics to display
        """
        print("\n" + "="*80)
        print("BACKTEST METRICS REPORT")
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
    """Main entry point for script."""
    if len(sys.argv) < 2:
        print("Usage: python backtest_metrics.py <strategy_id> [start_date] [end_date]")
        print("\nExample:")
        print("  python backtest_metrics.py buy3 2010-01-01 2023-12-31")
        sys.exit(1)

    strategy_id = sys.argv[1]
    start_date = sys.argv[2] if len(sys.argv) > 2 else None
    end_date = sys.argv[3] if len(sys.argv) > 3 else None

    # Initialize simulator
    historical_data_dir = Path(__file__).parent.parent / "data" / "stock_data"
    simulator = BacktestSimulator(str(historical_data_dir))

    # Run backtest
    try:
        trades, metrics = simulator.run_backtest(strategy_id, start_date, end_date)

        # Print report
        simulator.print_report(metrics)

        # Optionally save trades to CSV
        if trades:
            output_file = Path(__file__).parent.parent / "logs" / f"backtest_metrics_{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
