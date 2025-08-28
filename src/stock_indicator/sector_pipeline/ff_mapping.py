"""Functions for loading and applying Fama-French industry mappings.

The default mapping resides in ``data/sic_to_ff.csv``. Updated tables are
available from the Kenneth French Data Library and should replace this file
when industry definitions change.
"""

# TODO: review

import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def load_fama_french_mapping(source: str | Path) -> pd.DataFrame:
    """Load the Fama-French mapping table from ``source``.

    ``source`` may be a URL or a path to a CSV file.
    """
    source_str = str(source)
    if source_str.startswith("http://") or source_str.startswith("https://"):
        try:
            logger.info("Downloading Fama-French mapping from %s", source_str)
            response = requests.get(source_str, timeout=30)
            response.raise_for_status()
            mapping_data_frame = pd.read_csv(io.BytesIO(response.content))
        except requests.RequestException as error:
            logger.error("Failed to download mapping from %s: %s", source_str, error)
            raise
    else:
        mapping_path = Path(source_str)
        mapping_data_frame = pd.read_csv(mapping_path)
    mapping_data_frame.columns = [
        column.strip().lower() for column in mapping_data_frame.columns
    ]
    if "sic" in mapping_data_frame.columns and "sic_start" not in mapping_data_frame.columns:
        mapping_data_frame["sic_start"] = mapping_data_frame["sic"].astype(int)
        mapping_data_frame["sic_end"] = mapping_data_frame["sic"].astype(int)
    for column_name in ("ff12", "ff48", "ff49"):
        if column_name not in mapping_data_frame.columns:
            mapping_data_frame[column_name] = -1
    if "label" not in mapping_data_frame.columns:
        mapping_data_frame["label"] = ""
    mapping_data_frame["sic_start"] = mapping_data_frame["sic_start"].astype(int)
    mapping_data_frame["sic_end"] = mapping_data_frame["sic_end"].astype(int)
    return mapping_data_frame[["sic_start", "sic_end", "ff12", "ff48", "ff49", "label"]]


def build_classification_lookup(mapping_data_frame: pd.DataFrame) -> pd.DataFrame:
    """Expand SIC ranges into an explicit lookup table."""
    classification_rows: list[dict[str, Any]] = []
    for mapping_row in mapping_data_frame.itertuples(index=False):
        start_code = int(mapping_row.sic_start)
        end_code = int(mapping_row.sic_end)
        for classification_code in range(start_code, end_code + 1):
            classification_rows.append(
                {
                    "sic": classification_code,
                    "ff12": int(mapping_row.ff12),
                    "ff48": int(mapping_row.ff48),
                    "ff49": int(mapping_row.ff49),
                    "ff_label": mapping_row.label,
                }
            )
    return pd.DataFrame(classification_rows)


def attach_fama_french_groups(
    data_frame: pd.DataFrame, classification_lookup: pd.DataFrame
) -> pd.DataFrame:
    """Merge Fama-French classifications into ``data_frame`` using ``sic`` codes."""
    merged_data_frame = data_frame.merge(classification_lookup, on="sic", how="left")
    merged_data_frame["ff12"] = merged_data_frame["ff12"].fillna(-1).astype(int)
    merged_data_frame["ff48"] = merged_data_frame["ff48"].fillna(-1).astype(int)
    merged_data_frame["ff49"] = merged_data_frame["ff49"].fillna(-1).astype(int)
    merged_data_frame["ff_label"] = merged_data_frame["ff_label"].fillna("UNKNOWN")
    return merged_data_frame
