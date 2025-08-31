#!/bin/bash
set -euo pipefail

# ==== EDIT THESE TO MATCH YOUR MACHINE ====
REPO="$HOME/git_projects/stock_indicator"
SRC="$REPO/src"
VENV="$REPO/venv"

# Your daily_job argument line:
ARG_LINE='dollar_volume>2.14%,Top3 strategy=s1 0.1'

# ===== DO NOT EDIT BELOW UNLESS YOU KNOW WHY =====
LOGDIR="$REPO/cron_logs"
mkdir -p "$LOGDIR"

# Ensure we’re in the right place so -m can resolve the package
cd "$SRC"

# Run as a module so `from . import cron` works
# Stdout/stderr go to a rolling cron log for debugging
"$VENV/bin/python" -m stock_indicator.daily_job "$ARG_LINE" >> "$LOGDIR/cron_stdout.log" 2>&1
