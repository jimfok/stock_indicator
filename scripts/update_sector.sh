#!/bin/bash
# TODO: review

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
fi

# Allow caller to override in the environment.
export SEC_USER_AGENT="${SEC_USER_AGENT:-stock-indicator/1.0 (contact: maintainer@example.com)}"

LOG_FILE="$PROJECT_ROOT/logs/update_sector.log"
mkdir -p "$(dirname "$LOG_FILE")"
{
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting update"
    python -m stock_indicator.manage update_sector_data
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] finished with exit code $?"
} >> "$LOG_FILE" 2>&1