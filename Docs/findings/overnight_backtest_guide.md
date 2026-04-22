# Overnight Backtest Execution Guide

## Quick Instructions

### Run Overnight (Background with nohup)

```bash
cd ~/JimGit/stock_indicator
source .venv/bin/activate

# Run in background with nohup (survives terminal disconnect)
nohup python scripts/backtest_metrics_simple.py buy3 50 2010-01-01 2023-12-31 > logs/overnight_backtest.log 2>&1 &

# Save the PID
echo $! > logs/overnight_backtest.pid
echo "Backtest started in background. PID: $(cat logs/overnight_backtest.pid)"
echo "Monitor progress: tail -f logs/overnight_backtest.log"
```

### Check Progress Later

```bash
# Check if still running
if ps -p $(cat logs/overnight_backtest.pid) > /dev/null 2>&1; then
    echo "Backtest still running..."
else
    echo "Backtest completed"
fi

# View recent output
tail -100 logs/overnight_backtest.log
```

### Check Results When Done

```bash
# View the report
cat Docs/findings/backtest_metrics_buy3_top50_*.md

# Or view the log
cat logs/overnight_backtest.log | tail -50
```

## Scripts Available

### `scripts/backtest_metrics_simple.py` (RECOMMENDED)
- Uses existing `start_simulate` command (reliable)
- Parses console output to extract metrics
- Saves report to `Docs/findings/`
- Slower but accurate

### `scripts/backtest_metrics_fast.py`
- Directly calculates signals from scratch
- Faster in theory
- Currently has performance issues with volume profile calculations
- NOT recommended until optimized

## Expected Performance

Based on tests:
- **1 year, 10 symbols**: ~10-15 minutes
- **14 years (2010-2023), 10 symbols**: ~3-5 hours
- **14 years (2010-2023), 50 symbols**: ~8-12 hours

The backtest is computationally intensive because it calculates signals for all symbols for every day.

## Strategy: buy3

From `data/strategy_sets.csv`:
- **Buy**: `ema_sma_cross_testing_3_-99_99_-99.0,99.0_0.973,1.0`
- **Sell**: `ema_sma_cross_testing_3_-0.01_65_-10.0,10.0_0.78,1.00`

Interpretation:
- **EMA/SMA window**: 3 days
- **SMA angle**: unrestricted (-99° to 99°)
- **Near-price volume**: 97.3% to 100% (very tight filter)
- **Above-price volume**: unrestricted

This is a very aggressive, high-frequency strategy that should generate many trades.

## Trade Execution Rules

As per your request:

1. **Entry signal on date T** → Buy at market OPEN on T+1
2. **Exit signal on date T** → Sell at market OPEN on T+1

This matches the existing backtest system's behavior (confirmed by validation script).

## Notes

- The script automatically saves the results to `Docs/findings/backtest_metrics_*.md`
- Log output is saved to `logs/overnight_backtest.log`
- The PID file `logs/overnight_backtest.pid` helps you check if it's still running
- Use `nohup` to ensure the process continues even if you close your terminal
