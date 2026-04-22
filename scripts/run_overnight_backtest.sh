#!/bin/bash
# Quick start script for overnight backtest

# Usage: ./run_overnight_backtest.sh [strategy_id] [top_n] [start_date] [end_date]

STRATEGY_ID=${1:-buy3}
TOP_N=${2:-50}
START_DATE=${3:-2010-01-01}
END_DATE=${4:-2023-12-31}

echo "======================================================================="
echo "Starting Overnight Backtest"
echo "======================================================================="
echo "Strategy: $STRATEGY_ID"
echo "Top N: $TOP_N"
echo "Date Range: $START_DATE to $END_DATE"
echo ""

cd ~/JimGit/stock_indicator
source .venv/bin/activate

LOG_FILE="logs/overnight_backtest_${STRATEGY_ID}.log"
PID_FILE="logs/overnight_backtest_${STRATEGY_ID}.pid"

# Remove old log if exists
rm -f "$LOG_FILE"

# Start in background with nohup
nohup python scripts/backtest_metrics_simple.py "$STRATEGY_ID" "$TOP_N" "$START_DATE" "$END_DATE" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Backtest started in background."
echo "PID: $PID"
echo "PID file: $PID_FILE"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check if running:"
echo "  ps -p $PID"
echo ""
echo "Kill process (if needed):"
echo "  kill $PID"
echo "======================================================================="

# Save PID
echo "$PID" > "$PID_FILE"
