"""FTD + BSV breathing strategy research.

Strategy:
1. FTD triggers → enter at close
2. First white bar → exit
3. Next white bar (≥5 bars later) → re-enter
4. Next white bar (≥5 bars later) → exit
5. Repeat until SL or trend dies

Stop: each entry has a fixed SL (3%).
"Stop buying" rule: 2 consecutive losing rounds → stop for this FTD cycle.
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
SL_PCT = 0.03
MIN_WHITE_GAP = 5  # minimum bars between white bar signals
MAX_CYCLE_BARS = 120  # max bars per FTD cycle
CONSECUTIVE_LOSS_STOP = 2  # stop buying after N consecutive losses


def _ftd_signal_series(df: pandas.DataFrame) -> pandas.Series:
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


def simulate_symbol(symbol: str, df: pandas.DataFrame) -> list[dict]:
    bsv_df = bsv(df["high"], df["low"], df["close"], df["volume"])
    df = df.join(bsv_df)
    df["is_white"] = df["footprint_flag"] > 0

    ftd_signals = _ftd_signal_series(df)

    # Deduplicate FTD: skip if fired within last 20 bars
    ftd_dates = []
    last_ftd_loc = -999
    for i, (dt, val) in enumerate(ftd_signals.items()):
        if val and (i - last_ftd_loc) > 20:
            ftd_dates.append((i, dt))
            last_ftd_loc = i

    all_rounds = []

    for ftd_loc, ftd_date in ftd_dates:
        if ftd_loc + 10 >= len(df):
            continue

        # State machine: breathing in/out
        in_position = True
        entry_price = df["close"].iloc[ftd_loc]
        entry_loc = ftd_loc
        entry_bar = 0
        round_num = 0
        consecutive_losses = 0
        last_white_loc = ftd_loc  # treat FTD as "last event"
        cycle_end = min(ftd_loc + MAX_CYCLE_BARS, len(df))

        for loc in range(ftd_loc + 1, cycle_end):
            bar_offset = loc - ftd_loc
            current_price = df["close"].iloc[loc]
            current_low = df["low"].iloc[loc]
            is_white = df["is_white"].iloc[loc]

            if in_position:
                # Check SL
                sl_price = entry_price * (1 - SL_PCT)
                if current_low <= sl_price:
                    # SL hit
                    exit_price = sl_price
                    ret = (exit_price - entry_price) / entry_price
                    hold = loc - entry_loc
                    all_rounds.append({
                        "symbol": symbol,
                        "ftd_date": str(ftd_date.date()) if hasattr(ftd_date, 'date') else str(ftd_date),
                        "round": round_num,
                        "entry_bar": entry_loc - ftd_loc,
                        "exit_bar": bar_offset,
                        "hold": hold,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": ret * 100,
                        "exit_reason": "SL",
                    })
                    consecutive_losses += 1
                    in_position = False
                    last_white_loc = loc
                    round_num += 1

                    if consecutive_losses >= CONSECUTIVE_LOSS_STOP:
                        break
                    continue

                # Check white bar exit (must be ≥ MIN_WHITE_GAP from last event)
                if is_white and (loc - last_white_loc) >= MIN_WHITE_GAP:
                    exit_price = current_price
                    ret = (exit_price - entry_price) / entry_price
                    hold = loc - entry_loc
                    all_rounds.append({
                        "symbol": symbol,
                        "ftd_date": str(ftd_date.date()) if hasattr(ftd_date, 'date') else str(ftd_date),
                        "round": round_num,
                        "entry_bar": entry_loc - ftd_loc,
                        "exit_bar": bar_offset,
                        "hold": hold,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": ret * 100,
                        "exit_reason": "white_bar",
                    })
                    if ret > 0:
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1

                    in_position = False
                    last_white_loc = loc
                    round_num += 1

                    if consecutive_losses >= CONSECUTIVE_LOSS_STOP:
                        break

            else:
                # Not in position — look for next white bar to re-enter
                if is_white and (loc - last_white_loc) >= MIN_WHITE_GAP:
                    entry_price = current_price
                    entry_loc = loc
                    in_position = True
                    last_white_loc = loc

        # If still in position at cycle end, close at market
        if in_position and entry_loc < cycle_end - 1:
            exit_loc = min(cycle_end - 1, len(df) - 1)
            exit_price = df["close"].iloc[exit_loc]
            ret = (exit_price - entry_price) / entry_price
            hold = exit_loc - entry_loc
            all_rounds.append({
                "symbol": symbol,
                "ftd_date": str(ftd_date.date()) if hasattr(ftd_date, 'date') else str(ftd_date),
                "round": round_num,
                "entry_bar": entry_loc - ftd_loc,
                "exit_bar": exit_loc - ftd_loc,
                "hold": hold,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": ret * 100,
                "exit_reason": "cycle_end",
            })

    return all_rounds


def main():
    all_rounds = []
    for sym in UNIVERSE:
        LOGGER.info("Processing %s...", sym)
        try:
            df = yfinance.download(sym, start=START, end=END, auto_adjust=True, progress=False)
        except Exception:
            continue
        if isinstance(df.columns, pandas.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if len(df) < 200:
            continue
        rounds = simulate_symbol(sym, df)
        all_rounds.extend(rounds)

    df = pandas.DataFrame(all_rounds)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("FTD + BSV BREATHING STRATEGY (%d rounds across %d FTD cycles)",
                len(df), df.groupby(["symbol", "ftd_date"]).ngroups)
    LOGGER.info("=" * 80)

    # Overall stats
    wins = (df["return_pct"] > 0).sum()
    losses = (df["return_pct"] <= 0).sum()
    wr = wins / len(df) * 100
    mean_ret = df["return_pct"].mean()
    profits = df.loc[df["return_pct"] > 0, "return_pct"]
    loss_trades = df.loc[df["return_pct"] <= 0, "return_pct"]
    mp = profits.mean() if len(profits) > 0 else 0
    ml = loss_trades.mean() if len(loss_trades) > 0 else 0
    pl = abs(mp / ml) if ml != 0 else float("inf")

    LOGGER.info("\n--- Overall ---")
    LOGGER.info("  Rounds: %d  WR: %.1f%%  Mean: %.2f%%  MP: %.2f%%  ML: %.2f%%  P/L: %.2f",
                len(df), wr, mean_ret, mp, ml, pl)
    LOGGER.info("  Mean hold: %.1f bars  Median hold: %.1f bars",
                df["hold"].mean(), df["hold"].median())

    # By exit reason
    LOGGER.info("\n--- By exit reason ---")
    for reason, grp in df.groupby("exit_reason"):
        w = (grp["return_pct"] > 0).sum()
        r = w / len(grp) * 100
        LOGGER.info("  %-12s  n=%d  WR=%.1f%%  mean=%.2f%%  hold=%.1f bars",
                    reason, len(grp), r, grp["return_pct"].mean(), grp["hold"].mean())

    # By round number
    LOGGER.info("\n--- By round number (0=FTD entry, 1=first re-entry, ...) ---")
    for rnd, grp in df.groupby("round"):
        if len(grp) < 5:
            continue
        w = (grp["return_pct"] > 0).sum()
        r = w / len(grp) * 100
        LOGGER.info("  Round %d: n=%d  WR=%.1f%%  mean=%.2f%%  hold=%.1f bars",
                    rnd, len(grp), r, grp["return_pct"].mean(), grp["hold"].mean())

    # Per-cycle P/L (sum of all rounds in one FTD cycle)
    LOGGER.info("\n--- Per FTD cycle ---")
    cycles = df.groupby(["symbol", "ftd_date"]).agg(
        total_return=("return_pct", "sum"),
        num_rounds=("round", "max"),
        total_bars=("exit_bar", "max"),
    ).reset_index()
    cycle_wins = (cycles["total_return"] > 0).sum()
    LOGGER.info("  Cycles: %d  Profitable: %d (%.1f%%)",
                len(cycles), cycle_wins, cycle_wins / len(cycles) * 100)
    LOGGER.info("  Mean cycle return: %.2f%%  Median: %.2f%%",
                cycles["total_return"].mean(), cycles["total_return"].median())
    LOGGER.info("  Mean rounds per cycle: %.1f  Mean bars per cycle: %.1f",
                (cycles["num_rounds"] + 1).mean(), cycles["total_bars"].mean())

    # Per-symbol
    LOGGER.info("\n--- Per symbol ---")
    for sym in UNIVERSE:
        sym_df = df[df["symbol"] == sym]
        if len(sym_df) == 0:
            continue
        sym_cycles = cycles[cycles["symbol"] == sym]
        w = (sym_df["return_pct"] > 0).sum()
        r = w / len(sym_df) * 100
        LOGGER.info(
            "  %s: %d cycles, %d rounds, WR=%.0f%%, mean=%.2f%%, cycle_mean=%.2f%%",
            sym, len(sym_cycles), len(sym_df), r,
            sym_df["return_pct"].mean(),
            sym_cycles["total_return"].mean(),
        )

    # The big question: cumulative return
    LOGGER.info("\n--- Cumulative (simple sum, no compounding) ---")
    total = df["return_pct"].sum()
    LOGGER.info("  Total return: %.1f%% over %d rounds", total, len(df))
    LOGGER.info("  ppb (profit per bar held): %.3f%%", df["return_pct"].sum() / df["hold"].sum())

    csv_path = "data/ftd_bsv_breathing.csv"
    df.to_csv(csv_path, index=False)
    LOGGER.info("\nResults saved to %s", csv_path)


if __name__ == "__main__":
    main()
