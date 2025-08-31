"""Helpers for manual sector overrides.

This module allows assigning a default Fama–French industry group to symbols
that are missing from the automatically built sector classification dataset.
Overrides are stored in a simple CSV file at ``data/sector_overrides.csv``.
"""

# TODO: review

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_OUTPUT_PARQUET_PATH,
    DEFAULT_OUTPUT_CSV_PATH,
    DATA_DIRECTORY,
)

LOGGER = logging.getLogger(__name__)

SECTOR_OVERRIDES_CSV_PATH = DATA_DIRECTORY / "sector_overrides.csv"


def _load_sector_dataset() -> pd.DataFrame | None:
    """Return the latest sector dataset if available, otherwise ``None``.

    The returned DataFrame has lower-cased column names for consistency.
    """
    try:
        if DEFAULT_OUTPUT_PARQUET_PATH.exists():
            frame = pd.read_parquet(DEFAULT_OUTPUT_PARQUET_PATH)
        elif DEFAULT_OUTPUT_CSV_PATH is not None and DEFAULT_OUTPUT_CSV_PATH.exists():
            frame = pd.read_csv(DEFAULT_OUTPUT_CSV_PATH)
        else:
            return None
    except Exception:  # noqa: BLE001
        return None
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    return frame


def _load_overrides() -> pd.DataFrame:
    """Return the sector overrides file as a DataFrame with lower-cased columns.

    When the overrides file does not exist, an empty DataFrame with the
    expected ``ticker`` and ``ff12`` columns is returned.
    """
    if not SECTOR_OVERRIDES_CSV_PATH.exists():
        return pd.DataFrame(columns=["ticker", "ff12"])
    try:
        overrides = pd.read_csv(SECTOR_OVERRIDES_CSV_PATH)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["ticker", "ff12"])
    overrides.columns = [str(c).strip().lower() for c in overrides.columns]
    # Ensure expected columns exist
    if "ticker" not in overrides.columns:
        overrides["ticker"] = []  # type: ignore[assignment]
    if "ff12" not in overrides.columns:
        overrides["ff12"] = []  # type: ignore[assignment]
    # Normalize values
    overrides["ticker"] = overrides["ticker"].astype(str).str.upper()
    overrides["ff12"] = pd.to_numeric(overrides["ff12"], errors="coerce").astype("Int64")
    return overrides


def _save_overrides(overrides: pd.DataFrame) -> None:
    """Persist ``overrides`` to ``SECTOR_OVERRIDES_CSV_PATH``.

    The DataFrame must contain ``ticker`` and ``ff12`` columns.
    """
    SECTOR_OVERRIDES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write minimal columns to keep the file simple and stable
    columns_to_write = ["ticker", "ff12"]
    overrides = overrides.copy()
    overrides["ticker"] = overrides["ticker"].astype(str).str.upper()
    overrides["ff12"] = pd.to_numeric(overrides["ff12"], errors="coerce").astype("Int64")
    overrides[columns_to_write].sort_values("ticker").to_csv(
        SECTOR_OVERRIDES_CSV_PATH, index=False
    )


def assign_symbol_to_other_if_missing(symbol: str) -> bool:
    """Assign a symbol without sector data to the 'Other' group (FF12 = 12).

    If the symbol already appears in the sector dataset with an FF12 group,
    no change is made. When the main dataset lacks this symbol, an override is
    recorded in ``data/sector_overrides.csv`` with ``ff12`` set to 12.

    Returns ``True`` when an override was added or updated, ``False`` when
    nothing changed.

    Parameters
    ----------
    symbol: str
        The ticker symbol to check and override when needed.
    """
    normalized_symbol = (symbol or "").strip().upper()
    if not normalized_symbol:
        return False

    # Check if the symbol already exists in the main sector dataset
    sector_frame = _load_sector_dataset()
    if sector_frame is not None and "ticker" in sector_frame.columns:
        match = sector_frame[sector_frame["ticker"].astype(str).str.upper() == normalized_symbol]
        if not match.empty:
            # If the dataset explicitly tags this symbol, do not override
            return False

    overrides = _load_overrides()
    # Upsert the override with ff12 = 12
    existing_index = overrides.index[
        overrides["ticker"].astype(str).str.upper() == normalized_symbol
    ].tolist()
    if existing_index:
        # If already set to 12, nothing to do
        current_value = overrides.loc[existing_index[0], "ff12"]
        try:
            current_int = int(current_value) if pd.notna(current_value) else None
        except Exception:  # noqa: BLE001
            current_int = None
        if current_int == 12:
            return False
        overrides.loc[existing_index[0], "ff12"] = 12
        _save_overrides(overrides)
        LOGGER.info("Updated sector override: %s -> FF12=12", normalized_symbol)
        return True

    # Insert new override
    overrides = pd.concat(
        [
            overrides,
            pd.DataFrame({"ticker": [normalized_symbol], "ff12": [12]}),
        ],
        ignore_index=True,
    )
    _save_overrides(overrides)
    LOGGER.info("Added sector override: %s -> FF12=12", normalized_symbol)
    return True

