SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TMPL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"

SEC_USER_AGENT = "Your Name your_email@example.com"

CACHE_DIR = "cache"
SUBMISSIONS_DIR = f"{CACHE_DIR}/submissions"
LAST_RUN_CONFIG = f"{CACHE_DIR}/last_run.json"
DEFAULT_OUT_PARQUET = "data/symbols_with_sector.parquet"
DEFAULT_OUT_CSV = "data/symbols_with_sector.csv"

NORMALIZE_DOT_DASH = True
