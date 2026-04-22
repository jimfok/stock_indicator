#!/usr/bin/env python3
"""
Simple backtest metrics script using the existing start_simulate command.

This script calls the built-in backtest system and parses the output to get:
- Win rate
- Mean win percentage
- Trades per year

Usage:
    python backtest_metrics_simple.py <strategy_id> <top_n> [start_date] [end_date]

Example:
    python backtest_metrics_simple.py buy3 50 2010-01-01 2023-12-31
"""

import sys
import logging
from pathlib import Path
from subprocess import run, PIPE
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_backtest_output(output: str) -> dict:
    """
    Parse backtest console output to extract metrics.

    Args:
        output: Console output from start_simulate

    Returns:
        Dictionary with parsed metrics
    """
    metrics = {}

    # Extract summary line with: Trades: X, Win rate: Y%, Mean profit %: Z%, ...
    summary_match = re.search(
        r'Trades:\s*(\d+),\s*Win rate:\s*([\d.]+)%,\s*Mean profit %:\s*([\d.-]+)%,\s*Profit % Std Dev:\s*([\d.]+)%,\s*Mean loss %:\s*([\d.-]+)%',
        output
    )

    if summary_match:
        metrics['total_trades'] = int(summary_match.group(1))
        metrics['win_rate'] = float(summary_match.group(2))
        metrics['mean_profit_pct'] = float(summary_match.group(3))
        metrics['mean_loss_pct'] = float(summary_match.group(5))

    # Extract per-year statistics: "Year 2015: 15.23%, trade: 45"
    year_pattern = r'Year\s+(\d{4}):\s*([\d.-]+)%,\s*trade:\s*(\d+)'
    metrics['trades_by_year'] = {}

    for match in re.finditer(year_pattern, output):
        year = int(match.group(1))
        return_pct = float(match.group(2))
        trades = int(match.group(3))
        metrics['trades_by_year'][year] = trades

    # Extract final balance and CAGR
    balance_match = re.search(r'Final balance:\s*([\d.,]+)', output)
    if balance_match:
        # Remove commas from number
        balance_str = balance_match.group(1).replace(',', '')
        metrics['final_balance'] = float(balance_str)

    cagr_match = re.search(r'CAGR:\s*([\d.-]+)%', output)
    if cagr_match:
        metrics['cagr'] = float(cagr_match.group(1))

    drawdown_match = re.search(r'Max drawdown:\s*([\d.-]+)%', output)
    if drawdown_match:
        metrics['max_drawdown'] = float(drawdown_match.group(1))

    return metrics


def print_report(metrics: dict, strategy_id: str, top_n: int):
    """Print backtest metrics report."""
    print("\n" + "="*80)
    print(f"BACKTEST METRICS REPORT - {strategy_id} (Top {top_n})")
    print("="*80)

    if 'total_trades' in metrics:
        print(f"\nOverall Statistics:")
        print(f"  Total trades:       {metrics['total_trades']}")
        print(f"  Win rate:          {metrics['win_rate']:.2f}%")

        print(f"\nProfit/Loss:")
        print(f"  Mean profit %:     {metrics['mean_profit_pct']:+.4f}%")
        print(f"  Mean loss %:      {metrics['mean_loss_pct']:+.4f}%")

    if 'cagr' in metrics:
        print(f"\nPerformance:")
        print(f"  CAGR:              {metrics['cagr']:.2f}%")
    if 'max_drawdown' in metrics:
        print(f"  Max drawdown:      {metrics['max_drawdown']:.2f}%")

    if 'trades_by_year' in metrics:
        print(f"\nTrades Per Year:")
        for year in sorted(metrics['trades_by_year'].keys()):
            count = metrics['trades_by_year'][year]
            print(f"  {year}:              {count}")

    print("\n" + "="*80)


def run_backtest(strategy_id: str, top_n: int, start_date: str = None, end_date: str = None) -> dict:
    """
    Run backtest using the existing start_simulate command.

    Args:
        strategy_id: Strategy ID from strategy_sets.csv (e.g., 'buy3')
        top_n: Number of top symbols by dollar volume
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        Dictionary with parsed metrics
    """
    logger.info(f"Running backtest for strategy: {strategy_id}")
    logger.info(f"  Top N: {top_n}")
    if start_date:
        logger.info(f"  Start date: {start_date}")
    if end_date:
        logger.info(f"  End date: {end_date}")

    # Build command
    dollar_volume_filter = f"dollar_volume>0,{top_n}th"

    cmd = [
        sys.executable,
        "-m",
        "stock_indicator.manage",
        "start_simulate",
        f"start={start_date}" if start_date else "",
        f"dollar_volume>0,{top_n}th",
        "strategy=" + strategy_id,
        "1.0",  # stop loss
        "false"  # show_details
    ]

    # Remove empty strings
    cmd = [c for c in cmd if c]

    logger.info(f"Command: {' '.join(cmd)}")

    # Run command
    result = run(cmd, cwd=Path(__file__).parent.parent, capture_output=True, text=True)

    output = result.stdout
    error = result.stderr

    if result.returncode != 0:
        logger.error(f"Backtest failed with return code {result.returncode}")
        if error:
            logger.error(f"Error output:\n{error}")
        raise RuntimeError(f"Backtest failed")

    # Parse output
    metrics = parse_backtest_output(output)

    logger.info(f"Backtest complete")

    return metrics


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python backtest_metrics_simple.py <strategy_id> <top_n> [start_date] [end_date]")
        print("\nExample:")
        print("  python backtest_metrics_simple.py buy3 50 2010-01-01 2023-12-31")
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
        metrics = run_backtest(strategy_id, top_n, start_date, end_date)
        print_report(metrics, strategy_id, top_n)

        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = Path(__file__).parent.parent / "Docs" / "findings" / f"backtest_metrics_{strategy_id}_top{top_n}_{timestamp}.md"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        content = f"""# Backtest Metrics - {strategy_id} (Top {top_n})

## Summary

- **Strategy**: {strategy_id}
- **Top N symbols**: {top_n}
- **Date range**: {start_date or 'Oldest available'} to {end_date or 'Latest available'}
- **Generated**: {timestamp}

## Metrics

"""

        if 'total_trades' in metrics:
            content += f"""
| Metric | Value |
|---------|--------|
| Total trades | {metrics['total_trades']} |
| Win rate | {metrics['win_rate']:.2f}% |
| Mean profit % | {metrics['mean_profit_pct']:+.4f}% |
| Mean loss % | {metrics['mean_loss_pct']:+.4f}% |
"""

        if 'cagr' in metrics:
            content += f"| CAGR | {metrics['cagr']:.2f}% |\n"
        if 'max_drawdown' in metrics:
            content += f"| Max drawdown | {metrics['max_drawdown']:.2f}% |\n"

        if 'trades_by_year' in metrics:
            content += """
## Trades Per Year

| Year | Trades |
|------|--------|
"""
            for year in sorted(metrics['trades_by_year'].keys()):
                count = metrics['trades_by_year'][year]
                content += f"| {year} | {count} |\n"

        output_file.write_text(content)
        logger.info(f"Report saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
