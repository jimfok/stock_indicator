#!/bin/bash
# Run refined backtest with minimum holding period

# Usage: ./run_refined_overnight.sh <original_csv> <min_hold_days>

ORIGINAL_CSV=${1:-logs/simulate_result/simulation_20260421_194922.csv}
MIN_HOLD_DAYS=${2:-5}

LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/refined_backtest_min${MIN_HOLD_DAYS}_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/refined_backtest_min${MIN_HOLD_DAYS}_${TIMESTAMP}.pid"

echo "======================================================================="
echo "Running Refined Backtest - Minimum Holding Period"
echo "======================================================================="
echo "Original CSV:   $ORIGINAL_CSV"
echo "Min hold days:  $MIN_HOLD_DAYS"
echo "Log file:       $LOG_FILE"
echo ""

cd ~/JimGit/stock_indicator
source .venv/bin/activate

# Create logs directory
mkdir -p "$LOG_DIR"

# Run in background with nohup
nohup python scripts/run_refined_backtest.py "$ORIGINAL_CSV" "$MIN_HOLD_DAYS" > "$LOG_FILE" 2>&1 &
PID=$!

# Save PID
echo $PID > "$PID_FILE"

echo "Refined backtest started in background."
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
echo "Estimated time: ~30-60 minutes for 384 trades"
echo "======================================================================="
