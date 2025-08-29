"""Utilities for loading named strategy pairs from a config file.

The configuration lives at ``data/strategy_sets.csv`` and should contain
rows with columns: ``strategy_id,buy,sell``. Each row defines a reusable
strategy identifier that expands to a concrete buy/sell pair.

Example CSV contents:

    strategy_id,buy,sell
    default,ema_sma_cross_with_slope_40,ema_sma_cross_with_slope_50

"""

# TODO: review

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple


DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
STRATEGY_SET_PATH = DATA_DIRECTORY / "strategy_sets.csv"


def load_strategy_set_mapping(path: Path | None = None) -> Dict[str, Tuple[str, str]]:
    """Load a mapping of strategy_id to (buy, sell) strategy names.

    Returns an empty mapping when the file does not exist.
    """
    file_path = path or STRATEGY_SET_PATH
    if not file_path.exists():
        return {}
    mapping: Dict[str, Tuple[str, str]] = {}
    with file_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            strategy_id = (row.get("strategy_id") or "").strip()
            buy_name = (row.get("buy") or "").strip()
            sell_name = (row.get("sell") or "").strip()
            if not strategy_id or not buy_name or not sell_name:
                continue
            mapping[strategy_id] = (buy_name, sell_name)
    return mapping

