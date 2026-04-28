#!/bin/bash
set -euo pipefail

# Place TP/SL orders for today's filled BUY entries.
# Run ~10 minutes after US market open (09:40 ET = 21:40 HKT).
# Requires Futu OpenD to be running and logged in.

SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="${REPO:-$SCRIPT_DIRECTORY}"
SOURCE_DIRECTORY="${SRC:-$REPOSITORY_ROOT/src}"
VIRTUAL_ENVIRONMENT_DIRECTORY="${VENV:-$REPOSITORY_ROOT/venv}"
LOG_DIRECTORY="$REPOSITORY_ROOT/cron_logs"

mkdir -p "$LOG_DIRECTORY"
cd "$SOURCE_DIRECTORY"

"$VIRTUAL_ENVIRONMENT_DIRECTORY/bin/python" -m stock_indicator.place_tp_sl \
    >> "$LOG_DIRECTORY/tp_sl_stdout.log" 2>&1
