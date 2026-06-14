#!/usr/bin/env python3
"""
Batch backtest v2 — optimized with parallel Pass 1 + direct sell signal.

Optimizations:
  1. Sell signal computed directly (heavy EMA cross, no chip metrics)
  2. Parallel Pass 1 (multiprocessing) for CSV scanning
  3. Two-pass memory-efficient approach

Usage:
  python scripts/order_model_v2_batch.py --data 2025_2026 --start 2026-04-01 --end 2026-04-30
  python scripts/order_model_v2_batch.py --data 1989 --start 1989-01-01 --end 2025-12-31
"""

import argparse, gc, re, sys, time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stock_indicator.strategy import (
    BUY_STRATEGIES, load_price_data, parse_strategy_name, rename_signal_columns,
    sma, _build_eligibility_mask, _split_strategy_choices,
)

DATA_SOURCES = {
    "daily": PROJECT_ROOT / "data" / "stock_data",
    "2010":  PROJECT_ROOT / "data" / "stock_data_2010_yf_clean",
    "2014":  PROJECT_ROOT / "data" / "stock_data_2014",
    "1989":  PROJECT_ROOT / "data" / "stock_data_1989",
    "test1": PROJECT_ROOT / "data" / "stock_data_test1",
    "2025_2026": PROJECT_ROOT / "data" / "stock_data_2025_2026",
}
DV_SMA_WINDOW = 50
TP_PCT = 0.0722

# --- Multiprocessing worker (must be module-level) ---

_BUF = None
_END_TS = None
_DV_SMA_W = DV_SMA_WINDOW
_SCAN_DIR = None

def _init_worker(buf, end_ts, scan_dir):
    """Set module-level globals for workers."""
    global _BUF, _END_TS, _SCAN_DIR
    _BUF = buf
    _END_TS = end_ts
    _SCAN_DIR = scan_dir

def _scan_csv(stem: str) -> Optional[Tuple[str, pd.Series]]:
    """Read one CSV, compute DV SMA, return (symbol, series) or None."""
    global _BUF, _END_TS, _SCAN_DIR
    try:
        p = _SCAN_DIR / f"{stem}.csv"
        if not p.exists():
            return None
        raw = pd.read_csv(p, parse_dates=["Date"], index_col="Date",
                          usecols=lambda c: c in ("Date", "close", "volume"))
        raw = raw.loc[_BUF:_END_TS].copy()
        if raw.empty:
            return None
        s = sma(raw["close"] * raw["volume"], _DV_SMA_W).dropna()
        return (stem, s) if not s.empty else None
    except Exception:
        return None

# --- Helpers ---

def _bars_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return max(0, len(pd.bdate_range(start, end)) - 1)


def parse_filter(f: str) -> Tuple[Optional[float], Optional[int], int]:
    ratio = top = None
    pick = 1
    m = re.search(r",Pick(\d+)", f, re.IGNORECASE)
    if m: pick = int(m.group(1)); f = f[:m.start()] + f[m.end():]
    m = re.fullmatch(r"dollar_volume>(\d+(?:\.\d{1,2})?)%,Top(\d+)", f, re.IGNORECASE)
    if m: return float(m.group(1)) / 100.0, int(m.group(2)), pick
    m = re.fullmatch(r"dollar_volume>(\d+(?:\.\d{1,2})?)%,(\d+)th", f, re.IGNORECASE)
    if m: return float(m.group(1)) / 100.0, int(m.group(2)), pick
    m = re.fullmatch(r"dollar_volume>(\d+(?:\.\d+)?),Top(\d+)", f, re.IGNORECASE)
    if m: return float(m.group(1)), int(m.group(2)), pick
    m = re.fullmatch(r"dollar_volume>(\d+(?:\.\d{1,2})?)%", f, re.IGNORECASE)
    if m: return float(m.group(1)) / 100.0, top, pick
    raise ValueError(f"Bad filter: {f}")


def compute_exit_signal(df: pd.DataFrame, window_size: int = 3,
                         exit_alpha_factor: float = 3.0) -> pd.Series:
    """Compute heavy EMA cross exit signal — NO chip metrics needed."""
    close_r3 = df["close"].round(3)
    sma_vals = sma(close_r3, window_size)
    heavy_alpha = min(exit_alpha_factor / (window_size + 1), 1.0)
    ema_heavy = close_r3.ewm(alpha=heavy_alpha, adjust=False).mean()
    cross_down = (ema_heavy.shift(1) >= sma_vals.shift(1)) & (ema_heavy < sma_vals)
    return cross_down.shift(1, fill_value=False).fillna(False).astype(bool)


def simulate_symbol(df: pd.DataFrame, symbol: str,
                     entry_col: str, exit_col: str,
                     start_date: pd.Timestamp, end_date: pd.Timestamp,
                     tp_pct: float = TP_PCT, min_hold: int = 5,
                     max_cash: float = 6000.0) -> List[dict]:
    """Simulate per-symbol trades."""
    sim = df.loc[start_date:end_date]
    if sim.empty: return []
    sim = sim[~sim.index.duplicated(keep='first')]
    rows = list(sim.iterrows())

    trades = []
    in_pos = False
    sig_date = fill_price = fill_date = tp_trigger = 0.0
    shares = 0
    sl_price = None
    trail_high = 0.0
    prev_closes = []

    for i, (bar_date, bar) in enumerate(rows):
        open_p = float(bar["open"])
        high_p = float(bar.get("high", open_p))
        low_p = float(bar.get("low", open_p))
        close_p = float(bar.get("close", open_p))
        if "close" in bar: prev_closes.append(float(bar["close"]))

        if not in_pos and bar.get(entry_col, False):
            sh = int(max_cash / open_p)
            if sh > 0:
                shares = sh; fill_price = open_p; fill_date = bar_date
                sig_date = bar_date; tp_trigger = open_p * (1.0 + tp_pct)
                sl_price = None; trail_high = open_p; in_pos = True
            continue
        if not in_pos: continue

        # TP
        if high_p >= tp_trigger:
            ep = tp_trigger if open_p < tp_trigger else open_p
            trades.append({"symbol": symbol, "entry_date": fill_date, "exit_date": bar_date,
                           "signal_date": sig_date, "entry_price": fill_price, "exit_price": ep,
                           "shares": shares, "pnl": (ep - fill_price) * shares,
                           "commission": abs((ep - fill_price) * shares) * 0.001,
                           "holding_period": _bars_between(sig_date, bar_date),
                           "exit_reason": "take_profit"})
            in_pos = False; continue

        # SL
        bs = _bars_between(sig_date, bar_date)
        if bs >= min_hold:
            if sl_price is None:
                lk = min(20, len(prev_closes))
                sp = 0.02
                if lk >= 5:
                    r = pd.Series(prev_closes[-lk:])
                    atr = (r.max() - r.min()) / r.mean()
                    sp = min(0.03, max(0.01, atr * 2.0))
                sl_price = fill_price * (1.0 - sp)
            else:
                if close_p > trail_high: trail_high = close_p
                sl_price = max(sl_price, fill_price + (trail_high - fill_price) * 0.5)
            if low_p <= sl_price:
                ep = sl_price if open_p > sl_price else open_p
                trades.append({"symbol": symbol, "entry_date": fill_date, "exit_date": bar_date,
                               "signal_date": sig_date, "entry_price": fill_price, "exit_price": ep,
                               "shares": shares, "pnl": (ep - fill_price) * shares,
                               "commission": abs((ep - fill_price) * shares) * 0.001,
                               "holding_period": bs, "exit_reason": "stop_loss"})
                in_pos = False; continue

        # Exit signal
        if bar.get(exit_col, False) and bs >= min_hold:
            ep = open_p
            trades.append({"symbol": symbol, "entry_date": fill_date, "exit_date": bar_date,
                           "signal_date": sig_date, "entry_price": fill_price, "exit_price": ep,
                           "shares": shares, "pnl": (ep - fill_price) * shares,
                           "commission": abs((ep - fill_price) * shares) * 0.001,
                           "holding_period": bs, "exit_reason": "signal_exit"})
            in_pos = False; continue

    if in_pos:
        lb = rows[-1][1]
        ep = float(lb.get("close", lb["open"]))
        trades.append({"symbol": symbol, "entry_date": fill_date, "exit_date": rows[-1][0],
                       "signal_date": sig_date, "entry_price": fill_price, "exit_price": ep,
                       "shares": shares, "pnl": (ep - fill_price) * shares,
                       "commission": abs((ep - fill_price) * shares) * 0.001,
                       "holding_period": _bars_between(sig_date, rows[-1][0]),
                       "exit_reason": "forced_exit"})
    return trades


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="2025_2026", choices=list(DATA_SOURCES))
    p.add_argument("--start", default="2026-04-01")
    p.add_argument("--end", default="2026-04-30")
    p.add_argument("--filter", default="dollar_volume>0.02%,Top500,Pick5")
    p.add_argument("--buy", default="ema_sma_cross_testing_3_-99_99_-99.0,99.0_0.973,1.0")
    p.add_argument("--sell", default="ema_sma_cross_testing_3_-0.01_65_-10.0,10.0_0.78,1.00")
    p.add_argument("--tp", type=float, default=TP_PCT)
    p.add_argument("--min-hold", type=int, default=5)
    p.add_argument("--max-pos", type=int, default=6)
    p.add_argument("--cash", type=float, default=60000.0)
    p.add_argument("--pos-size", type=float, default=0.10)
    p.add_argument("--exit-alpha", type=float, default=3.0)
    p.add_argument("--log", action="store_true")
    p.add_argument("--workers", type=int, default=0,
                   help="Pass 1 workers (0 = use all CPUs)")
    args = p.parse_args()

    data_dir = DATA_SOURCES[args.data]
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    log = lambda msg: print(msg, file=sys.stderr) if args.log else None

    print(f"Data: {data_dir.name} {args.start}→{args.end}")
    print(f"Filter: {args.filter} TP:{args.tp*100:.1f}% Hold:{args.min_hold}")
    t0 = time.time()

    # PASS 1: Build merged volume frame (parallel)
    csv_stems = sorted([p.stem for p in data_dir.glob("*.csv") if not p.stem.startswith("^GSPC")])
    log(f"CSVs: {len(csv_stems)}")

    buf = start_ts - pd.Timedelta(days=100)
    n_workers = args.workers or cpu_count()
    log(f"Parallel scan with {n_workers} workers...")

    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(buf, end_ts, data_dir)) as pool:
        # imap_unordered for streaming results as they finish
        results = []
        for i, res in enumerate(pool.imap_unordered(_scan_csv, csv_stems, chunksize=200)):
            if res is not None:
                results.append(res)
            if args.log and (i + 1) % 2000 == 0:
                log(f"  scanned {i+1}/{len(csv_stems)} ({len(results)} ok) ({time.time()-t0:.0f}s)")

    log(f"Pass1: {len(results)} symbols ({time.time()-t0:.0f}s)")

    if not results:
        print("No symbols with volume data.")
        return

    mvol = pd.concat([s for _, s in results], axis=1, keys=[stem for stem, _ in results])
    mvol = mvol.sort_index()
    del results; gc.collect()
    log(f"MV: {mvol.shape}")

    min_r, top_r, pick = parse_filter(args.filter)
    mask = _build_eligibility_mask(mvol, minimum_average_dollar_volume=None,
        top_dollar_volume_rank=top_r, minimum_average_dollar_volume_ratio=min_r,
        maximum_symbols_per_group=pick)
    log(f"Mask: {mask.shape}")
    del mvol; gc.collect()

    vi = mask.index[mask.index >= start_ts]
    if not len(vi): print("No data"); return
    mr = mask.loc[vi[0]:end_ts]
    eligible = sorted([c for c in mr.columns if mr[c].any()])
    log(f"Eligible: {len(eligible)}")
    del mask; gc.collect()

    # Parse buy strategy
    buy_choices = _split_strategy_choices(args.buy)
    bf = []
    for n in buy_choices:
        try:
            base, w, ar, nr, ab = parse_strategy_name(n)
            if base in BUY_STRATEGIES: bf.append((n, base, w, ar, nr, ab))
        except: pass

    # Parse sell window
    sell_choices = _split_strategy_choices(args.sell)
    sell_window = 3
    for n in sell_choices:
        try:
            _, w, _, _, _ = parse_strategy_name(n)
            if w: sell_window = w
        except: pass
    log(f"Sell window: {sell_window}, exit_alpha: {args.exit_alpha}")

    # PASS 2: Per-symbol simulation
    all_trades = []
    sc = 0
    for sym in eligible:
        csv_path = data_dir / f"{sym}.csv"
        if not csv_path.exists(): continue
        try:
            df = load_price_data(csv_path)
        except: continue
        if df.empty: continue

        # Buy signals (full strategy function — includes chip metrics)
        buy_cols = []
        for rn, base, w, ar, nr, ab in bf:
            fn = BUY_STRATEGIES[base]
            kw = {}
            if w: kw["window_size"] = w
            if ar: kw["angle_range"] = ar
            if nr and ab: kw["near_range"] = nr; kw["above_range"] = ab
            try:
                fn(df, **kw)
                rename_signal_columns(df, base, rn)
                c = f"{rn}_entry_signal"
                if c in df.columns: buy_cols.append(c)
            except Exception as e:
                log(f"  ERR {sym} strategy: {e}")
                continue

        # Sell signal: DIRECT heavy EMA cross
        exit_sig = compute_exit_signal(df, sell_window, args.exit_alpha)

        df["_entry"] = df[buy_cols].any(axis=1).fillna(False) if buy_cols else False
        df["_exit"] = exit_sig

        if sym in mr.columns:
            df["_entry"] = df["_entry"] & mr[sym].reindex(df.index, fill_value=False)

        trades = simulate_symbol(df, sym, "_entry", "_exit",
            start_ts, end_ts, args.tp, args.min_hold, args.cash * args.pos_size)
        all_trades.extend(trades)
        sc += 1
        if args.log and sc % 200 == 0:
            log(f"Sim {sc}/{len(eligible)} {len(all_trades)} trades ({time.time()-t0:.0f}s)")

    log(f"Pass2: {sc} syms, {len(all_trades)} trades ({time.time()-t0:.0f}s)")

    # Portfolio cap: track open positions
    all_trades.sort(key=lambda t: t["entry_date"])
    acc = []
    ol = []
    for t in all_trades:
        ol = [o for o in ol if o["exit_date"] > t["entry_date"]]
        if len(ol) < args.max_pos: acc.append(t); ol.append(t)
    log(f"After cap: {len(acc)}")

    if not acc:
        print("No trades.")
        return

    # Cash sizing
    cash = args.cash
    rt = []
    for t in acc:
        sh = int(cash * args.pos_size / t["entry_price"])
        if sh <= 0: continue
        cost = sh * t["entry_price"]
        if cost > cash: continue
        cash -= cost
        pnl = (t["exit_price"] - t["entry_price"]) * sh
        rt.append({**t, "shares": sh, "pnl": pnl, "commission": abs(pnl) * 0.001})
        cash += sh * t["exit_price"]

    if not rt:
        print("No trades after cash.")
        return

    tp = sum(t["pnl"] for t in rt)
    tc = sum(t["commission"] for t in rt)
    w = sum(1 for t in rt if t["pnl"] > 0)
    l = len(rt) - w
    fe = args.cash + tp - tc

    print(f"\n{'='*60}")
    print(f"  Trades: {len(rt)}  ({w}W/{l}L, {w/len(rt)*100:.1f}% WR)")
    print(f"  Net P&L: ${tp-tc:>+9.2f}  Final: ${fe:>,.2f}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")

    # Save trades to CSV
    out_df = pd.DataFrame(rt)
    csv_path = PROJECT_ROOT / f"backtest_v2_trades_{args.data}_{args.start}_{args.end}.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"  Trades saved: {csv_path}")

    syms = {}
    for t in rt:
        syms.setdefault(t["symbol"], []).append(t)
    for s, ts in sorted(syms.items(), key=lambda x: -len(x[1])):
        print(f"  {s:6s}: {len(ts):3d} trades, ${sum(t['pnl'] for t in ts):>+8.2f}")

    reasons = {}
    for t in rt:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
    print(f"\n  Exit reasons:")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {r:15s}: {c}")


if __name__ == "__main__":
    main()
