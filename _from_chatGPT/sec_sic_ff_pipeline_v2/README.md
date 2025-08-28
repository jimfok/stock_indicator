# SEC→SIC→Fama‑French Classification Pipeline

This mini-project fetches SEC metadata to map U.S. tickers to CIK→SIC→Fama‑French (FF12/FF48/FF49).
Run nightly; uses local caching; only new/changed symbols are fetched.

Quick start
-----------
1) pip install -r requirements.txt
2) Set SEC_USER_AGENT in src/sec_pipeline/config.py
3) Build:
   python main.py build      --symbols-url https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt      --ff-map-url https://raw.githubusercontent.com/Wenzhi-Ding/FamaFrenchIndustry/main/ff_48ind.csv      --out data/symbols_with_sector.parquet
4) Outputs at data/symbols_with_sector.parquet and .csv
