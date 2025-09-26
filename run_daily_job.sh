#!/bin/bash
set -euo pipefail

# Allow the cron job to open more file descriptors
ulimit -n 4096

# Determine repository path from script location or override with environment variables
SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="${REPO:-$SCRIPT_DIRECTORY}"
SOURCE_DIRECTORY="${SRC:-$REPOSITORY_ROOT/src}"
VIRTUAL_ENVIRONMENT_DIRECTORY="${VENV:-$REPOSITORY_ROOT/.venv}"

# Your daily_job argument line
ARG_LINE_1='dollar_volume>0.05%,Top50,Pick2 strategy=s4 1.0'
ARG_LINE_2='dollar_volume>0.05%,Top20,Pick7 strategy=s6 1.0'

# Set up logging directories
LOG_DIRECTORY="$REPOSITORY_ROOT/cron_logs"
DATE_LOG_DIRECTORY="$REPOSITORY_ROOT/logs"
mkdir -p "$LOG_DIRECTORY" "$DATE_LOG_DIRECTORY"

# Ensure the module can be resolved
cd "$SOURCE_DIRECTORY"

# TODO: review - compute the latest trading date and one-year start date
LATEST_DATE="$("$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -c 'from stock_indicator.daily_job import determine_latest_trading_date as determine_date;print(determine_date())')"
START_DATE="$("$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -c 'from datetime import datetime, timedelta; import sys; print((datetime.fromisoformat(sys.argv[1]) - timedelta(days=365)).date().isoformat())' "$LATEST_DATE")"

# Update historical data and record signals
"$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage update_all_data_from_yf "$START_DATE" "$LATEST_DATE" >> "$LOG_DIRECTORY/cron_stdout.log" 2>&1

# TODO: review
{
  echo "$ARG_LINE_1"
  "$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage find_history_signal "$LATEST_DATE" "$ARG_LINE_1"
} | tee -a "$LOG_DIRECTORY/cron_stdout.log" >> "$DATE_LOG_DIRECTORY/$LATEST_DATE.log"

# TODO: review
{
  echo "$ARG_LINE_2"
  "$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.manage find_history_signal "$LATEST_DATE" "$ARG_LINE_2"
} | tee -a "$LOG_DIRECTORY/cron_stdout.log" >> "$DATE_LOG_DIRECTORY/$LATEST_DATE.log"
