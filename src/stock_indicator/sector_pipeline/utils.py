"""Utility helpers for file handling, rate limiting, and ticker normalization."""

# TODO: review

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path: Path) -> None:
    """Create ``directory_path`` if it does not already exist."""
    directory_path.mkdir(parents=True, exist_ok=True)


def save_json_file(data: Any, file_path: Path) -> None:
    """Write ``data`` as JSON to ``file_path``."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file_pointer:
        json.dump(data, file_pointer, ensure_ascii=False, indent=2)


def load_json_file(file_path: Path) -> Any:
    """Load and return JSON content from ``file_path``."""
    with file_path.open("r", encoding="utf-8") as file_pointer:
        return json.load(file_pointer)


def sleep_politely(seconds: float = 0.12) -> None:
    """Pause execution for ``seconds`` to respect remote API rate limits."""
    time.sleep(seconds)


def normalize_ticker_symbol(symbol: str, convert_dash_to_dot: bool = True) -> str:
    """Return an upper-case ticker symbol, converting dashes to dots if requested."""
    cleaned_symbol = (symbol or "").strip().upper()
    if convert_dash_to_dot:
        return re.sub(r"(?<=\w)-(?!$)", ".", cleaned_symbol)
    return cleaned_symbol
