"""Pipeline for tagging symbols with SIC and Fama-French groups.

The pipeline stores intermediate data in directories under ``cache/`` within
the project root. Each call to :func:`build_sector_classification_dataset`
records its configuration in ``cache/last_run.json`` and caches SEC submission
files in ``cache/submissions``. Subsequent executions can call
:func:`update_latest_dataset` to rebuild the output using that saved
configuration while reusing any cached submissions. This incremental approach
avoids downloading data for symbols that have already been processed.
"""

# TODO: review

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .config import (
    SUBMISSIONS_DIRECTORY,
    LAST_RUN_CONFIG_PATH,
    DEFAULT_OUTPUT_PARQUET_PATH,
    DEFAULT_OUTPUT_CSV_PATH,
    SIC_TO_FAMA_FRENCH_MAPPING_PATH,
)
from .utils import ensure_directory_exists, save_json_file, load_json_file
from .sec_api import map_tickers_to_central_index_and_classification
from .ff_mapping import (
    load_fama_french_mapping,
    build_classification_lookup,
    attach_fama_french_groups,
)

logger = logging.getLogger(__name__)

# Ensure required cache directories exist for incremental updates
ensure_directory_exists(LAST_RUN_CONFIG_PATH.parent)
ensure_directory_exists(SUBMISSIONS_DIRECTORY)


def load_universe(source: str | Path) -> pd.DataFrame:
    """Load a universe of ticker symbols from ``source``.

    ``source`` may be a CSV file with a ``ticker`` column or a newline-separated list
    of symbols provided as a file or URL.
    """
    source_str = str(source)
    if source_str.endswith(".csv"):
        data_frame = pd.read_csv(source_str)
        if "ticker" not in data_frame.columns:
            raise ValueError("CSV input must contain a 'ticker' column")
        return data_frame[["ticker"]].dropna().drop_duplicates()
    if source_str.startswith("http://") or source_str.startswith("https://"):
        try:
            logger.info("Downloading ticker universe from %s", source_str)
            response = requests.get(source_str, timeout=30)
            response.raise_for_status()
            text = response.text
        except requests.RequestException as error:
            logger.error("Failed to download ticker universe from %s: %s", source_str, error)
            raise
    else:
        with Path(source_str).open("r", encoding="utf-8") as file_pointer:
            text = file_pointer.read()
    tickers = [symbol.strip().upper() for symbol in text.splitlines() if symbol.strip()]
    return pd.DataFrame({"ticker": tickers})


def build_sector_classification_dataset(
    symbols_source: str | Path,
    mapping_source: str | Path = SIC_TO_FAMA_FRENCH_MAPPING_PATH,
    output_parquet_path: Path = DEFAULT_OUTPUT_PARQUET_PATH,
    output_csv_path: Optional[Path] = DEFAULT_OUTPUT_CSV_PATH,
) -> pd.DataFrame:
    """Generate a data set of symbols tagged with CIK, SIC, and Fama-French codes.

    ``mapping_source`` may be a path to a CSV file or a URL pointing to one.
    By default the local ``sic_to_ff.csv`` file under the repository's
    ``data`` directory is used.
    """
    ensure_directory_exists(LAST_RUN_CONFIG_PATH.parent)
    ensure_directory_exists(SUBMISSIONS_DIRECTORY)
    universe_data_frame = load_universe(symbols_source)
    ticker_mapping_data_frame = map_tickers_to_central_index_and_classification(
        universe_data_frame
    )
    mapping_data_frame = load_fama_french_mapping(mapping_source)
    lookup_data_frame = build_classification_lookup(mapping_data_frame)
    classified_data_frame = attach_fama_french_groups(
        ticker_mapping_data_frame, lookup_data_frame
    )
    classified_data_frame["sic_desc"] = ""
    ensure_directory_exists(output_parquet_path.parent)
    classified_data_frame.to_parquet(output_parquet_path, index=False)
    if output_csv_path is not None:
        ensure_directory_exists(output_csv_path.parent)
        classified_data_frame.to_csv(output_csv_path, index=False)
    save_json_file(
        {
            "symbols_source": str(symbols_source),
            "mapping_source": str(mapping_source),
            "output": str(output_parquet_path),
        },
        LAST_RUN_CONFIG_PATH,
    )
    return classified_data_frame


def update_latest_dataset() -> pd.DataFrame:
    """Rebuild the classification data using the most recent configuration."""
    configuration = load_json_file(LAST_RUN_CONFIG_PATH)
    return build_sector_classification_dataset(
        configuration["symbols_source"],
        configuration["mapping_source"],
        Path(configuration.get("output", DEFAULT_OUTPUT_PARQUET_PATH)),
    )


def generate_coverage_report(data_frame: pd.DataFrame) -> str:
    """Return coverage information for ``data_frame`` as a formatted string."""
    total_count = len(data_frame)
    with_cik = data_frame["cik"].notna().sum()
    with_sic = data_frame["sic"].notna().sum()
    with_fama_french = (data_frame["ff48"] != -1).sum()
    return (
        f"Total: {total_count}\n"
        f"CIK: {with_cik} ({with_cik / total_count:.1%})\n"
        f"SIC: {with_sic} ({with_sic / total_count:.1%})\n"
        f"FF tag: {with_fama_french} ({with_fama_french / total_count:.1%})"
    )
