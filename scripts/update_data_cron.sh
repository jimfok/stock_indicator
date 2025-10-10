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
DEFAULT_HISTORICAL_START_DATE="1990-01-01"
HISTORICAL_START_DATE="${HISTORICAL_START_DATE:-$DEFAULT_HISTORICAL_START_DATE}"
HISTORICAL_END_DATE="${HISTORICAL_END_DATE:-$(date -u +'%Y-%m-%d')}"

LOG_FILE="$PROJECT_ROOT/logs/update_data_pipeline.log"
mkdir -p "$(dirname "$LOG_FILE")"
{
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting sector update"
    python -m stock_indicator.manage update_sector_data
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting symbol refresh"
    python -m stock_indicator.manage update_symbols
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting historical download ($HISTORICAL_START_DATE -> $HISTORICAL_END_DATE)"
    python -m stock_indicator.manage update_all_data_from_yf "$HISTORICAL_START_DATE" "$HISTORICAL_END_DATE"
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] finished with exit code $?"
} >> "$LOG_FILE" 2>&1
