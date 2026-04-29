"""BSV regime change research.

Hypothesis: daily footprint_flag (|buy_ratio - sell_ratio| < 0.1) indicates
institutional presence, and is followed by a regime change — the market
switches from trending to sideways or vice versa.

Measurement:
- Pre-window: 10 bars before flagged bar
- Post-window: 10 bars after flagged bar
- Compare directional consistency (trending vs choppy) before/after
- Compare absolute returns before/after
- Compute regime flip rate: flagged vs non-flagged baseline
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

LOOKBACK = 10  # bars before/after
START = "2014-01-01"
END = "2026-04-28"


def _directional_ratio(returns: pandas.Series) -> float:
    """Fraction of return explained by direction. 1 = pure trend, 0 = choppy."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    cum = returns.sum()
    path = returns.abs().sum()
    if path == 0:
        return 0.0
    return abs(cum) / path  # 1 = all same sign, 0 = cancels out


def analyze_symbol(symbol: str) -> dict | None:
    LOGGER.info("Processing %s...", symbol)
    try:
        df = yfinance.download(symbol, start=START, end=END, auto_adjust=True, progress=False)
    except Exception as e:
        LOGGER.warning("  Failed to download %s: %s", symbol, e)
        return None

    if isinstance(df.columns, pandas.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if len(df) < 100:
        return None

    bsv_df = bsv(df["high"], df["low"], df["close"], df["volume"])
    df = df.join(bsv_df)
    df["daily_return"] = df["close"].pct_change()

    flagged_idx = df.index[df["footprint_flag"] > 0]
    non_flagged_idx = df.index[df["footprint_flag"] == 0]

    results = {"symbol": symbol, "total_bars": len(df), "flagged_bars": len(flagged_idx)}

    # For each group, compute pre/post directional ratio and check for flip
    def _compute_regime_stats(indices, label):
        pre_dir_ratios = []
        post_dir_ratios = []
        flips = 0
        valid = 0
        pre_abs_returns = []
        post_abs_returns = []

        for dt in indices:
            loc = df.index.get_loc(dt)
            if loc < LOOKBACK or loc + LOOKBACK >= len(df):
                continue

            pre_rets = df["daily_return"].iloc[loc - LOOKBACK:loc]
            post_rets = df["daily_return"].iloc[loc + 1:loc + 1 + LOOKBACK]

            if len(pre_rets) < LOOKBACK or len(post_rets) < LOOKBACK:
                continue

            pre_dr = _directional_ratio(pre_rets)
            post_dr = _directional_ratio(post_rets)
            pre_dir_ratios.append(pre_dr)
            post_dir_ratios.append(post_dr)

            pre_abs_returns.append(pre_rets.abs().mean())
            post_abs_returns.append(post_rets.abs().mean())

            # Flip = trending→choppy or choppy→trending
            # Threshold: 0.5 for trending vs choppy
            pre_trending = pre_dr > 0.5
            post_trending = post_dr > 0.5
            if pre_trending != post_trending:
                flips += 1
            valid += 1

        if valid == 0:
            return {}

        return {
            f"{label}_n": valid,
            f"{label}_pre_dir_ratio": numpy.mean(pre_dir_ratios),
            f"{label}_post_dir_ratio": numpy.mean(post_dir_ratios),
            f"{label}_dir_ratio_change": numpy.mean(post_dir_ratios) - numpy.mean(pre_dir_ratios),
            f"{label}_flip_rate": flips / valid,
            f"{label}_pre_abs_ret": numpy.mean(pre_abs_returns),
            f"{label}_post_abs_ret": numpy.mean(post_abs_returns),
        }

    # Sample non-flagged to keep runtime reasonable
    rng = numpy.random.default_rng(42)
    sample_size = min(len(non_flagged_idx), len(flagged_idx) * 3)
    non_flagged_sample = rng.choice(non_flagged_idx, size=sample_size, replace=False)

    flagged_stats = _compute_regime_stats(flagged_idx, "flagged")
    baseline_stats = _compute_regime_stats(non_flagged_sample, "baseline")

    results.update(flagged_stats)
    results.update(baseline_stats)
    return results


def main():
    all_results = []
    for sym in UNIVERSE:
        r = analyze_symbol(sym)
        if r:
            all_results.append(r)

    df = pandas.DataFrame(all_results)

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("BSV REGIME CHANGE ANALYSIS (lookback=%d bars)", LOOKBACK)
    LOGGER.info("=" * 70)

    for _, row in df.iterrows():
        LOGGER.info(
            "\n%s  (flagged=%d / %d = %.1f%%)",
            row["symbol"],
            row.get("flagged_bars", 0),
            row.get("total_bars", 0),
            row.get("flagged_bars", 0) / max(row.get("total_bars", 1), 1) * 100,
        )
        if "flagged_flip_rate" in row and not pandas.isna(row["flagged_flip_rate"]):
            LOGGER.info(
                "  Flagged:  flip_rate=%.1f%%  dir_ratio %.3f → %.3f  (Δ%+.3f)  abs_ret %.4f → %.4f",
                row["flagged_flip_rate"] * 100,
                row["flagged_pre_dir_ratio"],
                row["flagged_post_dir_ratio"],
                row["flagged_dir_ratio_change"],
                row["flagged_pre_abs_ret"],
                row["flagged_post_abs_ret"],
            )
        if "baseline_flip_rate" in row and not pandas.isna(row["baseline_flip_rate"]):
            LOGGER.info(
                "  Baseline: flip_rate=%.1f%%  dir_ratio %.3f → %.3f  (Δ%+.3f)  abs_ret %.4f → %.4f",
                row["baseline_flip_rate"] * 100,
                row["baseline_pre_dir_ratio"],
                row["baseline_post_dir_ratio"],
                row["baseline_dir_ratio_change"],
                row["baseline_pre_abs_ret"],
                row["baseline_post_abs_ret"],
            )

    # Aggregate
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("AGGREGATE")
    LOGGER.info("=" * 70)
    for col in ["flip_rate", "dir_ratio_change", "pre_abs_ret", "post_abs_ret"]:
        f_col = f"flagged_{col}"
        b_col = f"baseline_{col}"
        if f_col in df.columns and b_col in df.columns:
            f_mean = df[f_col].mean()
            b_mean = df[b_col].mean()
            LOGGER.info(
                "  %-20s  flagged=%.4f  baseline=%.4f  diff=%+.4f",
                col, f_mean, b_mean, f_mean - b_mean,
            )

    csv_path = "data/bsv_regime_change_results.csv"
    df.to_csv(csv_path, index=False)
    LOGGER.info("\nResults saved to %s", csv_path)


if __name__ == "__main__":
    main()
