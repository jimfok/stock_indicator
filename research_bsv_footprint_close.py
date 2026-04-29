"""BSV footprint close research.

Idea: only look at closes on footprint (white) bars — days where
buy/sell power is balanced (institutional presence). Connect these
closes to see the "institutional price path" vs the full price path.

Analysis:
1. Overlay: footprint closes vs all closes
2. Trend clarity: directional ratio of footprint-only returns vs all returns
3. Predictive: does the footprint close trend predict future full-price direction?
4. Turning points: do footprint closes mark local extremes?
"""

from __future__ import annotations

import logging
import sys

import numpy
import pandas
import yfinance

sys.path.insert(0, "src")
from stock_indicator.indicators import bsv

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "JNJ", "PG", "CL", "ABBV", "KO", "PEP", "MRK", "UNH", "HD",
    "WMT", "XOM", "CVX",
]

START = "2014-01-01"
END = "2026-04-28"


def _directional_ratio(returns: pandas.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    path = returns.abs().sum()
    if path == 0:
        return 0.0
    return abs(returns.sum()) / path


def analyze_symbol(symbol: str) -> dict | None:
    LOGGER.info("Processing %s...", symbol)
    try:
        df = yfinance.download(symbol, start=START, end=END, auto_adjust=True, progress=False)
    except Exception as e:
        LOGGER.warning("  Failed: %s", e)
        return None

    if isinstance(df.columns, pandas.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if len(df) < 200:
        return None

    bsv_df = bsv(df["high"], df["low"], df["close"], df["volume"])
    df = df.join(bsv_df)

    flagged = df[df["footprint_flag"] > 0].copy()

    if len(flagged) < 20:
        return None

    # --- 1. Trend clarity ---
    all_returns = df["close"].pct_change().dropna()
    # Footprint-to-footprint returns (close to next footprint close)
    fp_returns = flagged["close"].pct_change().dropna()

    # Rolling window directional ratio (50-bar windows for all, proportional for fp)
    all_dir = _directional_ratio(all_returns)
    fp_dir = _directional_ratio(fp_returns)

    # Rolling directional ratio in chunks
    chunk_size_all = 50
    chunk_size_fp = max(5, len(flagged) // (len(df) // chunk_size_all))

    all_chunks = [all_returns.iloc[i:i+chunk_size_all]
                  for i in range(0, len(all_returns) - chunk_size_all, chunk_size_all)]
    fp_chunks = [fp_returns.iloc[i:i+chunk_size_fp]
                 for i in range(0, len(fp_returns) - chunk_size_fp, chunk_size_fp)]

    all_dir_chunks = [_directional_ratio(c) for c in all_chunks]
    fp_dir_chunks = [_directional_ratio(c) for c in fp_chunks]

    # --- 2. Predictive power ---
    # Does footprint close direction predict next 5/10 bars of full price?
    correct_5 = 0
    correct_10 = 0
    total_pred = 0

    for i in range(1, len(flagged)):
        fp_direction = flagged["close"].iloc[i] - flagged["close"].iloc[i - 1]
        loc = df.index.get_loc(flagged.index[i])

        if loc + 10 >= len(df):
            continue

        future_5 = df["close"].iloc[loc + 5] - df["close"].iloc[loc]
        future_10 = df["close"].iloc[loc + 10] - df["close"].iloc[loc]

        if fp_direction != 0:
            total_pred += 1
            if numpy.sign(fp_direction) == numpy.sign(future_5):
                correct_5 += 1
            if numpy.sign(fp_direction) == numpy.sign(future_10):
                correct_10 += 1

    # --- 3. Turning point analysis ---
    # Is a footprint bar more likely to be near a local high/low?
    window = 10
    df["local_high"] = df["close"] == df["close"].rolling(2 * window + 1, center=True).max()
    df["local_low"] = df["close"] == df["close"].rolling(2 * window + 1, center=True).min()
    df["local_extreme"] = df["local_high"] | df["local_low"]

    fp_extreme_rate = df.loc[flagged.index, "local_extreme"].mean()
    all_extreme_rate = df["local_extreme"].mean()

    # --- 4. Volatility comparison ---
    all_vol = all_returns.std()
    fp_vol = fp_returns.std()

    return {
        "symbol": symbol,
        "total_bars": len(df),
        "flagged_bars": len(flagged),
        "flagged_pct": len(flagged) / len(df) * 100,
        "all_dir_ratio": all_dir,
        "fp_dir_ratio": fp_dir,
        "all_dir_ratio_chunked": numpy.nanmean(all_dir_chunks),
        "fp_dir_ratio_chunked": numpy.nanmean(fp_dir_chunks),
        "all_volatility": all_vol,
        "fp_volatility": fp_vol,
        "vol_ratio": fp_vol / all_vol if all_vol > 0 else float("nan"),
        "pred_n": total_pred,
        "pred_5d_accuracy": correct_5 / total_pred if total_pred > 0 else float("nan"),
        "pred_10d_accuracy": correct_10 / total_pred if total_pred > 0 else float("nan"),
        "fp_extreme_rate": fp_extreme_rate,
        "all_extreme_rate": all_extreme_rate,
        "extreme_enrichment": fp_extreme_rate / all_extreme_rate if all_extreme_rate > 0 else float("nan"),
    }


def main():
    results = []
    for sym in UNIVERSE:
        r = analyze_symbol(sym)
        if r:
            results.append(r)

    df = pandas.DataFrame(results)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("BSV FOOTPRINT CLOSE ANALYSIS")
    LOGGER.info("=" * 80)

    for _, row in df.iterrows():
        LOGGER.info(
            "\n%s  (footprint bars: %d / %d = %.1f%%)",
            row["symbol"], row["flagged_bars"], row["total_bars"], row["flagged_pct"],
        )
        LOGGER.info(
            "  Trend clarity (chunked dir_ratio):  all=%.3f  footprint=%.3f  (%.1fx)",
            row["all_dir_ratio_chunked"], row["fp_dir_ratio_chunked"],
            row["fp_dir_ratio_chunked"] / row["all_dir_ratio_chunked"] if row["all_dir_ratio_chunked"] > 0 else 0,
        )
        LOGGER.info(
            "  Volatility:  all=%.4f  footprint=%.4f  (ratio=%.2fx)",
            row["all_volatility"], row["fp_volatility"], row["vol_ratio"],
        )
        LOGGER.info(
            "  Predictive (fp direction → future):  5d=%.1f%%  10d=%.1f%%  (n=%d)",
            row["pred_5d_accuracy"] * 100, row["pred_10d_accuracy"] * 100, row["pred_n"],
        )
        LOGGER.info(
            "  Turning points:  fp_extreme=%.1f%%  all_extreme=%.1f%%  enrichment=%.2fx",
            row["fp_extreme_rate"] * 100, row["all_extreme_rate"] * 100, row["extreme_enrichment"],
        )

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("AGGREGATE (mean across %d symbols)", len(df))
    LOGGER.info("=" * 80)
    LOGGER.info("  Footprint bar frequency:    %.1f%%", df["flagged_pct"].mean())
    LOGGER.info("  Trend clarity (chunked):    all=%.3f  fp=%.3f  (%.2fx)",
                df["all_dir_ratio_chunked"].mean(), df["fp_dir_ratio_chunked"].mean(),
                df["fp_dir_ratio_chunked"].mean() / df["all_dir_ratio_chunked"].mean())
    LOGGER.info("  Volatility ratio (fp/all):  %.2fx", df["vol_ratio"].mean())
    LOGGER.info("  Predictive 5d accuracy:     %.1f%%", df["pred_5d_accuracy"].mean() * 100)
    LOGGER.info("  Predictive 10d accuracy:    %.1f%%", df["pred_10d_accuracy"].mean() * 100)
    LOGGER.info("  Turning point enrichment:   %.2fx", df["extreme_enrichment"].mean())

    csv_path = "data/bsv_footprint_close_results.csv"
    df.to_csv(csv_path, index=False)
    LOGGER.info("\nResults saved to %s", csv_path)


if __name__ == "__main__":
    main()
