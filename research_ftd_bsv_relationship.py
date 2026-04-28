"""FTD + BSV relationship research.

Questions:
1. After FTD triggers, how often/when do white bars appear?
2. Is the frequency different from baseline (non-FTD periods)?
3. If you enter at FTD and exit at first white bar (with profit), what's the P/L?
4. Compare exit strategies: first white bar, first white bar with profit, signal exit.
"""

from __future__ import annotations

import logging
import sys

import numpy
import pandas
import yfinance

sys.path.insert(0, "src")
from stock_indicator.indicators import bsv, ema

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "JNJ", "PG", "CL", "ABBV", "KO", "PEP", "MRK", "UNH", "HD",
    "WMT", "XOM", "CVX",
]

START = "2014-01-01"
END = "2026-04-28"
MAX_HOLD = 60  # max bars to hold after FTD entry


def _ftd_signal_series(df: pandas.DataFrame) -> pandas.Series:
    """Vectorized FTD signal — returns boolean Series."""
    close = df["close"]
    low = df["low"]
    volume = df["volume"]

    ema_close = close.ewm(span=50, adjust=False).mean()
    ema_vol = volume.ewm(span=50, adjust=False).mean()

    rolling_low = low.rolling(window=23, min_periods=1).min()
    four_day_vol = volume.rolling(window=4).sum()

    ma_check = close.shift(3) < ema_close.shift(3)
    bottom_check = (rolling_low - low.shift(3)).abs() <= 1e-8
    low_check = (
        (low > low.shift(1))
        & (low.shift(1) > low.shift(2))
        & (low.shift(2) > low.shift(3))
    )
    vol_check = four_day_vol > four_day_vol.shift(3)
    vol_ema_check = (volume > ema_vol).rolling(window=7, min_periods=1).max().astype(bool)

    return ma_check & bottom_check & low_check & vol_check & vol_ema_check


def analyze_symbol(symbol: str) -> list[dict]:
    LOGGER.info("Processing %s...", symbol)
    try:
        df = yfinance.download(symbol, start=START, end=END, auto_adjust=True, progress=False)
    except Exception as e:
        LOGGER.warning("  Failed: %s", e)
        return []

    if isinstance(df.columns, pandas.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if len(df) < 200:
        return []

    # Compute indicators
    bsv_df = bsv(df["high"], df["low"], df["close"], df["volume"])
    df = df.join(bsv_df)
    df["is_white"] = df["footprint_flag"] > 0

    ftd_signals = _ftd_signal_series(df)

    # Deduplicate: only keep first FTD in a cluster (skip if FTD fired within last 10 bars)
    ftd_dates = []
    last_ftd_loc = -999
    for i, (dt, val) in enumerate(ftd_signals.items()):
        if val and (i - last_ftd_loc) > 10:
            ftd_dates.append((i, dt))
            last_ftd_loc = i

    trades = []
    for entry_loc, entry_date in ftd_dates:
        if entry_loc + MAX_HOLD >= len(df):
            continue

        entry_price = df["close"].iloc[entry_loc]

        # Track white bar appearances after entry
        white_bars_after = []
        first_white_with_profit = None
        first_white_any = None

        for offset in range(1, MAX_HOLD + 1):
            loc = entry_loc + offset
            if loc >= len(df):
                break

            current_price = df["close"].iloc[loc]
            pct_return = (current_price - entry_price) / entry_price
            is_white = df["is_white"].iloc[loc]

            if is_white:
                white_bars_after.append(offset)
                if first_white_any is None:
                    first_white_any = offset
                if first_white_with_profit is None and pct_return > 0:
                    first_white_with_profit = offset

        # Exit strategies
        # 1. Fixed hold (20 bars)
        fixed_20_loc = min(entry_loc + 20, len(df) - 1)
        fixed_20_ret = (df["close"].iloc[fixed_20_loc] - entry_price) / entry_price

        # 2. First white bar (any)
        if first_white_any is not None:
            wa_loc = entry_loc + first_white_any
            white_any_ret = (df["close"].iloc[wa_loc] - entry_price) / entry_price
        else:
            white_any_ret = None

        # 3. First white bar with profit
        if first_white_with_profit is not None:
            wp_loc = entry_loc + first_white_with_profit
            white_profit_ret = (df["close"].iloc[wp_loc] - entry_price) / entry_price
        else:
            white_profit_ret = None

        # 4. Max return within MAX_HOLD (best case reference)
        future_closes = df["close"].iloc[entry_loc + 1:entry_loc + MAX_HOLD + 1]
        max_ret = ((future_closes / entry_price) - 1).max() if len(future_closes) > 0 else 0

        trades.append({
            "symbol": symbol,
            "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
            "white_bar_count": len(white_bars_after),
            "first_white_any_bar": first_white_any,
            "first_white_profit_bar": first_white_with_profit,
            "fixed_20_ret": fixed_20_ret,
            "white_any_ret": white_any_ret,
            "white_profit_ret": white_profit_ret,
            "max_ret_60": max_ret,
        })

    return trades


def main():
    all_trades = []
    for sym in UNIVERSE:
        trades = analyze_symbol(sym)
        all_trades.extend(trades)

    df = pandas.DataFrame(all_trades)

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("FTD + BSV RELATIONSHIP ANALYSIS (%d FTD entries)", len(df))
    LOGGER.info("=" * 70)

    # 1. White bar timing after FTD
    LOGGER.info("\n--- White bar timing after FTD entry ---")
    LOGGER.info("  First white bar (any):        median=%s bars, mean=%.1f bars",
                df["first_white_any_bar"].median(),
                df["first_white_any_bar"].mean())
    LOGGER.info("  First white bar (with profit): median=%s bars, mean=%.1f bars",
                df["first_white_profit_bar"].dropna().median(),
                df["first_white_profit_bar"].dropna().mean())
    LOGGER.info("  White bars in 60-bar window:   mean=%.1f, median=%.1f",
                df["white_bar_count"].mean(),
                df["white_bar_count"].median())
    LOGGER.info("  %% of FTD with no white bar in 60 bars: %.1f%%",
                (df["first_white_any_bar"].isna().sum() / len(df)) * 100)

    # 2. Distribution of first white bar timing
    LOGGER.info("\n--- First white bar timing distribution ---")
    bins = [1, 3, 5, 10, 20, 30, 60]
    wa = df["first_white_any_bar"].dropna()
    for i in range(len(bins) - 1):
        count = ((wa >= bins[i]) & (wa < bins[i + 1])).sum()
        LOGGER.info("  Bar %d-%d: %d (%.1f%%)", bins[i], bins[i + 1] - 1, count, count / len(wa) * 100)

    # 3. Exit strategy comparison
    LOGGER.info("\n--- Exit strategy comparison ---")

    strategies = [
        ("Fixed 20 bars", df["fixed_20_ret"]),
        ("First white bar (any)", df["white_any_ret"].dropna()),
        ("First white bar (profit)", df["white_profit_ret"].dropna()),
    ]
    for name, rets in strategies:
        wins = (rets > 0).sum()
        wr = wins / len(rets) * 100 if len(rets) > 0 else 0
        mean_r = rets.mean() * 100
        median_r = rets.median() * 100
        profits = rets[rets > 0]
        losses = rets[rets <= 0]
        mp = profits.mean() * 100 if len(profits) > 0 else 0
        ml = losses.mean() * 100 if len(losses) > 0 else 0
        pl = abs(mp / ml) if ml != 0 else float("inf")
        LOGGER.info(
            "  %-30s  n=%d  WR=%.1f%%  mean=%.2f%%  median=%.2f%%  MP=%.2f%%  ML=%.2f%%  P/L=%.2f",
            name, len(rets), wr, mean_r, median_r, mp, ml, pl,
        )

    # Reference
    LOGGER.info("  %-30s  mean=%.2f%%  (best possible in 60 bars)",
                "Max return (reference)", df["max_ret_60"].mean() * 100)

    # 4. Per-symbol breakdown
    LOGGER.info("\n--- Per-symbol: first white bar (with profit) ---")
    for sym in UNIVERSE:
        sym_df = df[df["symbol"] == sym]
        if len(sym_df) == 0:
            continue
        wp = sym_df["white_profit_ret"].dropna()
        if len(wp) == 0:
            LOGGER.info("  %s: %d FTD entries, no white-bar-with-profit exits", sym, len(sym_df))
            continue
        wr = (wp > 0).sum() / len(wp) * 100
        LOGGER.info(
            "  %s: %d FTD, %d white exits, WR=%.0f%%, mean=%.2f%%, hold=%.1f bars",
            sym, len(sym_df), len(wp), wr, wp.mean() * 100,
            sym_df["first_white_profit_bar"].dropna().mean(),
        )

    csv_path = "data/ftd_bsv_relationship.csv"
    df.to_csv(csv_path, index=False)
    LOGGER.info("\nResults saved to %s", csv_path)


if __name__ == "__main__":
    main()
