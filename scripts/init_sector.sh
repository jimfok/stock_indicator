#!/bin/bash
# TODO: review

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") MAPPING_SOURCE [OUTPUT_PATH]" >&2
    echo "Example: $(basename "$0") https://example.com/sic_to_ff.csv" >&2
    exit 1
fi

MAPPING_SOURCE="$1"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_OUTPUT_PATH="$PROJECT_ROOT/data/symbols_with_sector.parquet"
OUTPUT_PATH="${2:-$DEFAULT_OUTPUT_PATH}"

cd "$PROJECT_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
fi

export SEC_USER_AGENT="${SEC_USER_AGENT:-stock-indicator/1.0 (contact: maintainer@example.com)}"

python -m stock_indicator.manage update_sector_data --ff-map-url="$MAPPING_SOURCE" "$OUTPUT_PATH"
