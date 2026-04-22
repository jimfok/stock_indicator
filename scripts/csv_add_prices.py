"""Cross-reference simulation CSV with trade_details log to add real buy/sell prices.

Usage:
    python scripts/csv_add_prices.py logs/simulate_result/simulation_20260421_194922.csv
    python scripts/csv_add_prices.py logs/simulate_result/simulation_20260421_194922.csv --detail logs/trade_detail/trade_details_20260421_194922.log
"""

import re
import sys
from pathlib import Path


def parse_log(detail_log_path: Path) -> dict[tuple, float]:
    """Parse trade_details log into a dict keyed by (date, symbol, side).

    Lines look like:
      2010-03-29 (1) EXC open 16.55 ...
      2010-04-01 (0) EXC close 16.74 ... win 1.13% signal
    """
    entries = {}
    # Pattern: date, pos_idx, symbol, side, price
    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2})\s+\((\d+)\)\s+(\S+)\s+(open|close)\s+([\d.]+)"
    )
    for line in detail_log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            date, pos_idx, symbol, side, price = m.groups()
            key = (date, symbol, side)
            entries[key] = float(price)
    return entries


def add_prices(csv_path: Path, detail_log_path: Path | None = None) -> None:
    if detail_log_path is None:
        # Derive log path from csv path
        stem = csv_path.stem.replace("simulation_", "")
        detail_log_path = csv_path.parent.parent / "trade_detail" / f"trade_details_{stem}.log"
        if not detail_log_path.exists():
            stem = csv_path.stem.replace("simulation_", "trade_details_")
            detail_log_path = csv_path.parent.parent / "trade_detail" / f"{stem}.log"

    print(f"Reading log: {detail_log_path}")
    price_map = parse_log(detail_log_path)
    print(f"Parsed {len(price_map)} price entries from log")

    import pandas as pd

    df = pd.read_csv(csv_path)
    buy_prices = []
    sell_prices = []

    for _, row in df.iterrows():
        entry_key = (str(row["entry_date"]), row["symbol"], "open")
        exit_key = (str(row["exit_date"]), row["symbol"], "close")
        buy_prices.append(price_map.get(entry_key))
        sell_prices.append(price_map.get(exit_key))

    df["buy_price"] = buy_prices
    df["sell_price"] = sell_prices

    out_path = csv_path.with_stem(csv_path.stem + "_with_prices")
    df.to_csv(out_path, index=False)
    print(f"Written: {out_path}")

    # Preview
    print(df[["entry_date", "symbol", "buy_price", "sell_price", "exit_date"]].head(10).to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add real buy/sell prices to simulation CSV")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--detail", type=Path, default=None)
    args = parser.parse_args()

    add_prices(args.csv, args.detail)