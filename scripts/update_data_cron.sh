#!/bin/bash
# TODO: review

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
fi

# Allow overrides so operators can supply their own identification.
export SEC_USER_AGENT="${SEC_USER_AGENT:-stock-indicator/1.0 (contact: maintainer@example.com)}"

# Historical range can be overridden via environment variables.
# Default run targets the prior calendar year for weekly maintenance.
CURRENT_YEAR_UTC="$(date -u +'%Y')"
PRIOR_YEAR="$((CURRENT_YEAR_UTC - 1))"
DEFAULT_ROLLING_START_DATE="${PRIOR_YEAR}-01-01"
DEFAULT_HISTORICAL_START_DATE="1990-01-01" # Use for backtesting only when explicitly set.
HISTORICAL_START_DATE="${HISTORICAL_START_DATE:-$DEFAULT_ROLLING_START_DATE}"
HISTORICAL_END_DATE="${HISTORICAL_END_DATE:-$(date -u +'%Y-%m-%d')}"
if [[ "$HISTORICAL_START_DATE" == "$DEFAULT_ROLLING_START_DATE" ]]; then
    RANGE_MODE_MESSAGE="weekly mode: refreshing data from the prior calendar year. Set HISTORICAL_START_DATE=1990-01-01 for full backtest coverage."
else
    RANGE_MODE_MESSAGE="custom range requested starting at $HISTORICAL_START_DATE. Full-history backfill uses HISTORICAL_START_DATE=1990-01-01."
fi

LOG_FILE="$PROJECT_ROOT/logs/update_data_pipeline.log"
mkdir -p "$(dirname "$LOG_FILE")"
{
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $RANGE_MODE_MESSAGE"
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting sector update"
    python -m stock_indicator.manage update_sector_data
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting symbol refresh"
    python -m stock_indicator.manage update_symbols
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting historical download ($HISTORICAL_START_DATE -> $HISTORICAL_END_DATE)"
    python -m stock_indicator.manage update_all_data_from_yf "$HISTORICAL_START_DATE" "$HISTORICAL_END_DATE"
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] finished with exit code $?"
} >> "$LOG_FILE" 2>&1
