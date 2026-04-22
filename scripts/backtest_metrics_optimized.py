#!/usr/bin/env python3
"""
Calculate backtest metrics - optimized version that filters symbols first.

Two-pass approach:
1. First pass: Scan all symbols to find which ones have entry signals
2. Second pass: Run backtest only on symbols with entry signals

Usage:
    python backtest_metrics_optimized.py <strategy_id> <top_n> [start_date] [end_date]

Example:
    python backtest_metrics_optimized.py buy3 50 2010-01-01 2023-12-31
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


class SymbolScanner:
    """Scan symbols to find which ones have entry signals."""

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

    def scan_for_entry_signals(
        self,
        symbols: List[str],
        buy_strategy: str,
        start_date: str,
        end_date: str
    ) -> Set[str]:
        """
        Scan symbols to find which ones have at least one entry signal.

        Returns:
            Set of symbol names that have entry signals
        """
        logger.info(f"Scanning {len(symbols)} symbols for entry signals...")

        # Sample dates to check (check more frequently for short window strategies)
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        # Check every month for short-term strategies, every 3 months for long-term
        check_dates = pd.date_range(start, end, freq='MS')  # Every month

        symbols_with_entries = set()

        for i, symbol in enumerate(symbols):
            if (i + 1) % 500 == 0:
                logger.info(f"  Scanned {i + 1}/{len(symbols)} symbols ({len(symbols_with_entries)} with entries so far)")

            df = self.load_historical_data(symbol)
            if df is None or len(df) < 10:
                continue

            try:
                # Check each sample date for entry signals
                for check_date in check_dates:
                    if check_date not in df.index:
                        continue

                    try:
                        signals = find_history_signal(
                            date_string=check_date.strftime('%Y-%m-%d'),
                            dollar_volume_filter="dollar_volume>0",  # No filter during scan
                            buy_strategy=buy_strategy,
                            sell_strategy="ignored",
                            stop_loss=1.0
                        )

                        entry_signals = signals.get('entry_signals', [])
                        if entry_signals:
                            symbols_with_entries.add(symbol)
                            break  # Found entry, no need to check other dates

                    except Exception as e:
                        logger.debug(f"Error scanning {symbol} at {check_date}: {e}")
                        continue

            except Exception as e:
                logger.debug(f"Error processing symbol {symbol}: {e}")
                continue

        logger.info(f"Scan complete: {len(symbols_with_entries)} symbols have entry signals")
        return symbols_with_entries


class BacktestSimulator:
    """Simulates trades using historical data."""

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
        start_date: str = None,
        end_date: str = None,
        symbols_to_trade: Set[str] = None
    ) -> Tuple[List[Trade], BacktestMetrics]:
        """Run backtest simulation for a strategy."""
        logger.info(f"Running backtest for strategy: {strategy_id}")

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
            # Use oldest data date
            all_dates = []
            for csv_file in self.historical_data_directory.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, parse_dates=['Date'], usecols=['Date'])
                    if not df.empty:
                        all_dates.append(df['Date'].min())
                except:
                    continue

            start_date = min(all_dates).strftime('%Y-%m-%d') if all_dates else '2010-01-01'

        if end_date is None:
            # Use latest data date
            all_dates = []
            for csv_file in self.historical_data_directory.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, parse_dates=['Date'], usecols=['Date'])
                    if not df.empty:
                        all_dates.append(df['Date'].max())
                except:
                    continue

            end_date = max(all_dates).strftime('%Y-%m-%d') if all_dates else datetime.now().strftime('%Y-%m-%d')

        logger.info(f"  Date range: {start_date} to {end_date}")

        # Load symbol list
        symbol_files = list(self.historical_data_directory.glob("*.csv"))
        all_symbols = [f.stem for f in symbol_files if f.stem != "^GSPC"]

        # Filter symbols if provided
        if symbols_to_trade:
            logger.info(f"  Using {len(symbols_to_trade)} pre-filtered symbols")
            symbols_to_process = symbols_to_trade
        else:
            # Calculate dollar volumes for all symbols
            logger.info(f"  Calculating 50-day dollar volume for {len(all_symbols)} symbols...")
            dollar_volumes = {}

            for symbol in all_symbols:
                df = self.load_historical_data(symbol)
                if df is None or len(df) < 50:
                    continue

                df['dollar_volume'] = df['close'] * df['volume']
                dv_50d = df['dollar_volume'].rolling(50).mean().iloc[-1]
                dollar_volumes[symbol] = dv_50d

            # Select top N by dollar volume
            sorted_symbols = sorted(dollar_volumes.items(), key=lambda x: x[1], reverse=True)
            symbols_to_process = [s[0] for s in sorted_symbols[:top_n]]

            logger.info(f"  Selected top {len(symbols_to_process)} symbols by 50-day dollar volume")

        # Simulate trades day by day
        trades: List[Trade] = []
        open_positions: Dict[str, Tuple[str, float]] = {}

        current_date = pd.to_datetime(start_date)
        end_date_parsed = pd.to_datetime(end_date)

        day_count = 0
        total_days = (end_date_parsed - current_date).days

        logger.info(f"Simulating trades for {len(symbols_to_process)} symbols...")
        logger.info(f"  Expected {total_days} days to process")

        # Sample every 3 months for signal checking (speed optimization)
        check_dates = pd.date_range(current_date, end_date_parsed, freq='3MS')

        while current_date <= end_date_parsed:
            day_count += 1
            if day_count % 100 == 0:
                logger.info(f"  Processing {current_date.strftime('%Y-%m-%d')} ({day_count}/{total_days})")

            current_date_str = current_date.strftime('%Y-%m-%d')

            # Only check for signals on sample dates
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

                    # Filter to only our symbols
                    entry_signals = entry_signals.intersection(symbols_to_process)
                    exit_signals = exit_signals.intersection(symbols_to_process)

                    # Process exit signals first (sell at next day's open)
                    for symbol in list(open_positions.keys()):
                        if symbol in exit_signals:
                            entry_date, entry_price = open_positions[symbol]
                            exit_trade_date, exit_price = self.get_next_trading_day_open(symbol, current_date_str)

                            if exit_trade_date and exit_price:
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

                except Exception as e:
                    logger.debug(f"Error getting signals for {current_date_str}: {e}")

            # Move to next day
            current_date += timedelta(days=1)

        # Close any remaining open positions
        if open_positions:
            logger.warning(f"Closing {len(open_positions)} remaining positions at end date")
            for symbol, (entry_date, entry_price) in open_positions.items():
                df = self.load_historical_data(symbol)
                if df is not None:
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

    def print_report(self, metrics: BacktestMetrics, strategy_id: str, top_n: int):
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
        print("Usage: python backtest_metrics_optimized.py <strategy_id> <top_n> [start_date] [end_date]")
        print("\nExample:")
        print("  python backtest_metrics_optimized.py buy3 50 2010-01-01 2023-12-31")
        print("\nParameters:")
        print("  strategy_id - Strategy ID from strategy_sets.csv (e.g., buy3, s3)")
        print("  top_n      - Number of top symbols by dollar volume to trade")
        print("  start_date - Optional start date (YYYY-MM-DD)")
        print("  end_date   - Optional end date (YYYY-MM-DD)")
        print("\nOptimization:")
        print("  Script scans symbols first to find which ones have entry signals")
        print("  Then runs backtest only on those symbols")
        print("  This dramatically reduces processing time")
        sys.exit(1)

    strategy_id = sys.argv[1]
    top_n = int(sys.argv[2])
    start_date = sys.argv[3] if len(sys.argv) > 3 else None
    end_date = sys.argv[4] if len(sys.argv) > 4 else None

    # Initialize
    data_dir = Path(__file__).parent.parent / "data" / "stock_data"
    scanner = SymbolScanner(str(data_dir))

    try:
        # Phase 1: Scan for symbols with entry signals
        print("\n" + "="*80)
        print("PHASE 1: Scanning symbols for entry signals")
        print("="*80)

        # Get all symbols
        symbol_files = list(data_dir.glob("*.csv"))
        all_symbols = [f.stem for f in symbol_files if f.stem != "^GSPC"]

        # Load strategy
        strategy_file = Path(__file__).parent.parent / "data" / "strategy_sets.csv"
        strategy_df = pd.read_csv(strategy_file)
        strategy_row = strategy_df[strategy_df['strategy_id'] == strategy_id]
        if strategy_row.empty:
            logger.error(f"Strategy '{strategy_id}' not found")
            sys.exit(1)

        buy_strategy = strategy_row.iloc[0]['buy']

        # Scan for entry signals (sample every 3 months)
        symbols_with_entries = scanner.scan_for_entry_signals(
            all_symbols,
            buy_strategy,
            start_date or '2010-01-01',
            end_date or datetime.now().strftime('%Y-%m-%d')
        )

        if not symbols_with_entries:
            logger.error("No symbols with entry signals found!")
            sys.exit(1)

        logger.info(f"\nPhase 1 complete: Found {len(symbols_with_entries)} symbols with entry signals")

        # Phase 2: Run backtest on filtered symbols
        print("\n" + "="*80)
        print("PHASE 2: Running backtest on filtered symbols")
        print("="*80)

        simulator = BacktestSimulator(str(data_dir))
        trades, metrics = simulator.run_backtest(
            strategy_id=strategy_id,
            top_n=top_n,
            start_date=start_date,
            end_date=end_date,
            symbols_to_trade=symbols_with_entries  # Use only symbols with entries
        )

        # Print report
        simulator.print_report(metrics, strategy_id, len(symbols_with_entries))

        # Save trades to CSV
        if trades:
            output_file = Path(__file__).parent.parent / "logs" / f"backtest_{strategy_id}_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
