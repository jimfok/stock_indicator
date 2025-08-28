"""Configuration settings for the sector classification pipeline.

The pipeline requires a table that maps standard industrial classification
codes to Fama-French industry groups. Download the latest mapping from the
Kenneth French Data Library at
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html and
save it as ``sic_to_ff.csv`` in the repository's ``data`` directory to keep the
classifications current.
"""

# TODO: review

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
SEC_USER_AGENT = "StockIndicatorApp contact@stockindicator.com"

CACHE_DIRECTORY = REPOSITORY_ROOT / "cache"
SUBMISSIONS_DIRECTORY = CACHE_DIRECTORY / "submissions"
LAST_RUN_CONFIG_PATH = CACHE_DIRECTORY / "last_run.json"

DATA_DIRECTORY = REPOSITORY_ROOT / "data"
SIC_TO_FAMA_FRENCH_MAPPING_PATH = DATA_DIRECTORY / "sic_to_ff.csv"
DEFAULT_OUTPUT_PARQUET_PATH = DATA_DIRECTORY / "symbols_with_sector.parquet"
DEFAULT_OUTPUT_CSV_PATH = DATA_DIRECTORY / "symbols_with_sector.csv"

NORMALIZE_DOT_TO_DASH = True
