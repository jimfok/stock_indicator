#!/bin/bash
# Overnight backtest runner using optimized start_simulate command

# This script uses the built-in start_simulate with dollar_volume filtering
# which is already optimized to only process top N symbols

# Usage: ./run_backtest_overnight.sh <strategy_id> <top_n> <start_date> <end_date>

STRATEGY_ID=${1:-buy3}
TOP_N=${2:-50}
START_DATE=${3:-2010-01-01}
END_DATE=${4:-2023-12-31}

LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/backtest_overnight_${STRATEGY_ID}_top${TOP_N}_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/backtest_overnight_${STRATEGY_ID}_top${TOP_N}_${TIMESTAMP}.pid"

echo "======================================================================="
echo "Starting Overnight Backtest - Optimized Version"
echo "======================================================================="
echo "Strategy:      $STRATEGY_ID"
echo "Top N symbols: $TOP_N"
echo "Date range:    $START_DATE to $END_DATE"
echo "Log file:      $LOG_FILE"
echo ""

cd ~/JimGit/stock_indicator
source .venv/bin/activate

# Create logs directory
mkdir -p "$LOG_DIR"

# Build command with dollar_volume filter to only process top N symbols
# This is MUCH faster than scanning all 11,851 symbols
COMMAND="python -m stock_indicator.manage start_simulate start=${START_DATE} dollar_volume>0,${TOP_N}th strategy=${STRATEGY_ID} 1.0 false"

echo "Command: $COMMAND"
echo ""

# Run in background with nohup
nohup $COMMAND > "$LOG_FILE" 2>&1 &
PID=$!

# Save PID
echo $PID > "$PID_FILE"

echo "Backtest started in background."
echo "PID:       $PID"
echo "PID file:  $PID_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check if running:"
echo "  ps -p $PID"
echo ""
echo "Kill process (if needed):"
echo "  kill $PID"
echo ""
echo "======================================================================="
echo "Estimated time for buy3 strategy: 1-2 hours for Top 50"
echo "Check back in the morning!"
echo "======================================================================="
