"""Configuration settings for the sector classification pipeline."""

# TODO: review

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
SEC_USER_AGENT = "Your Name your_email@example.com"

CACHE_DIRECTORY = REPOSITORY_ROOT / "cache"
SUBMISSIONS_DIRECTORY = CACHE_DIRECTORY / "submissions"
LAST_RUN_CONFIG_PATH = CACHE_DIRECTORY / "last_run.json"

DATA_DIRECTORY = REPOSITORY_ROOT / "data"
DEFAULT_OUTPUT_PARQUET_PATH = DATA_DIRECTORY / "symbols_with_sector.parquet"
DEFAULT_OUTPUT_CSV_PATH = DATA_DIRECTORY / "symbols_with_sector.csv"

NORMALIZE_DOT_TO_DASH = True
