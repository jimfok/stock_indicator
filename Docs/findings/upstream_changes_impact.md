# Upstream Changes: Impact on Order Model

**Date:** 2026-05-03
**Source:** CalGit (inventor upstream) → JimGit (your fork)
**Scope:** Filter expansion, SL/min_hold redesign, and downstream conflicts

---

## 1. Filter Universe: Top200 → Top500

### What Changed
- Dollar volume threshold lowered: `0.05%` → `0.02%`
- Universe expanded: `Top200` → `Top500`
- Pick5 per FF12 group stays same

### Already Applied ✅
| File | Status |
|------|--------|
| `run_daily_job.sh` `ARG_LINE_1` | ✅ `"dollar_volume>0.02%,Top500,Pick5"` |
| `data/multi_bucket_buy3_production.json` | ✅ `"dollar_volume>0.02%,Top500,Pick5"` |
| `manage.py`, `strategy.py`, `cron.py` (parameterized) | ✅ (accept value at call time) |

### Still on Top200 ❌
| File | Line | Value |
|------|------|-------|
| `scripts/order_model_v2.py` | 113 | `FILTER_STR = "dollar_volume>0.05%,Top200,Pick5"` |
| `scripts/order_model_v2.py` | 236 | `_get_top_symbols_for_date(..., top_n: int = 200)` |
| `scripts/order_model_v2.py` | 406 | `fast_signal_scan(... top_n=200, ...)` |
| `scripts/order_model_v2_batch.py` | 201 | `--filter default="...Top200..."` |
| `scripts/backtest_1989.py` | 42 | hardcoded Top200 |
| `scripts/backtest_2014.py` | 54 | hardcoded Top200 |
| `scripts/grid_search_sector_tp.py` | 28 | hardcoded Top200 |
| `test_rdt*.py`, `debug_signal2.py`, `debug_count_signals.py`, `test_signal_scan.py` | various | `top_dollar_volume_rank=200` |
| All JSON backtest configs except production | various | still `0.05%,Top200,Pick5` |

**Impact**: Backtests run with `order_model_v2.py` or the batch script use a smaller universe (Top200) than production (Top500). Results are inconsistent — if you're comparing backtest metrics against daily production signals, the symbol pool differs.

---

## 2. Stop-Loss Redesign: T+1 Placement, No min_hold Block

### Old Behavior (current code)
```
SL order  → placed at T+1, but dormant until T + min_hold_bars
Exit signal → blocked until min_hold_bars satisfied
```

### New Behavior (upstream design)
```
SL order  → placed at T+1, ACTIVE immediately from T+1
Exit signal → still blocked by min_hold_bars
```

### What This Changes

**In `scripts/order_model_v2.py`:**
- Line 998-1010: `sl_activation = signal_date + MIN_HOLD_BARS` → delete this, set `sl_activation = t1_date` (same as order_placed date)
- Line 747-758: SL fill check uses `bar_date >= position.sl_active_date` → becomes always-true since sl_activation = t1_date
- Removal simplifies: SL trigger logic is just `low <= trigger_price` from T+1 onward

**In `src/stock_indicator/place_tp_sl.py` (live trading):**
- Lines 246-273: SL placement currently deferred until `bars_held >= MIN_HOLD_BARS`
- New behavior: place SL IMMEDIATELY on T+1, no bars_held check
- `MIN_HOLD_BARS` still needed for exit signal blocking (step 5)

**In `src/stock_indicator/order_engine.py`:**
- `create_buy_order()` (lines 433-446): SL order does NOT set `sl_activation` → **SL is already active immediately**, which is correct per new behavior
- `_try_fill_sl()` (lines 230-231): checks `sl_activation` exists before blocking → already correct
- `evaluate_bar()` (lines 142-148): signal exit correctly checks `bars_held >= min_hold_bars` → already correct

### Key Finding: Engine vs Script Mismatch

The `order_engine.py` (library) already implements the new behavior correctly:
- SL active immediately (no `sl_activation` set → not checked)
- Exit signal blocked by `min_hold_bars`

But `order_model_v2.py` (script) adds its own `sl_activation` logic on top that **reintroduces the min_hold block on SL**. This means:

| Component | SL Blocked by min_hold? | Exit Signal Blocked by min_hold? |
|-----------|------------------------|----------------------------------|
| `order_engine.py` | ❌ (correct) | ✅ (correct) |
| `order_model_v2.py` | ✅ (WRONG) | ✅ (correct) |
| `place_tp_sl.py` | ✅ (WRONG) | N/A (exit signal handled elsewhere) |
| Upstream design | ❌ | ✅ |

**Fix needed in `order_model_v2.py`:**
- Line 1001-1002: Change `sl_activation_idx = sig_idx + MIN_HOLD_BARS` to `sl_activation_idx = t1_idx` (or just don't set `sl_activation` at all, matching `order_engine.py` behavior)

**Fix needed in `place_tp_sl.py`:**
- Remove or bypass lines 268-273 (bars_held < MIN_HOLD_BARS → SL deferred)
- Place SL immediately if position exists and no SL order is already open

---

## 3. Adaptive TP/SL Config: Override Behavior

The production config `multi_bucket_buy3_production.json` has:
```json
"override_min_hold_tp_only": true,
"min_hold_tp": 1
```

This means:
- **TP**: Overrides adaptive TP if it's below `min_hold_tp` (1 bar = immediately active) — but only for TP, not SL
- **SL**: Gets the computed adaptive value, no min_hold override applied
- Original `MIN_HOLD_BARS = 5` still applies to exit signal blocking

Your `order_model_v2.py` already reads these from config (lines 184-185), so the override logic is present in config loading but needs to be verified:
- Line 120: `MIN_HOLD_TP = 1` — matches config default
- Line 121: `ADAPTIVE_MIN_TP = 0.02` — matches config
- Line 124: `ADAPTIVE_FIXED_SL = 0.03` — matches config

---

## 4. `_bars_between` Implementation Divergence

Two different implementations exist across the codebase:

| Location | Implementation | Used For |
|----------|---------------|----------|
| `order_engine.py` line 361 | `(end - start).days` (calendar) | Holding period calculation |
| `order_model_v2.py` line 563 | `(end - start).days` (calendar) | Holding period calculation |
| `order_model_v2_batch.py` line 82 | `len(pd.bdate_range(start, end)) - 1` (business days) | Holding period calculation |
| `place_tp_sl.py` line 264 | `len(pd.bdate_range(entry_ts, today_ts)) - 1` (business days) | bars_held for min_hold check |

**Conflict**: Same function name `_bars_between`, different semantics. Calendar days gives different results than business days for the same date range (e.g., Fri→Mon = 3 calendar vs 1 business day). This means:
- `order_engine.py`/`order_model_v2.py` report longer holding periods
- `order_model_v2_batch.py`/`place_tp_sl.py` report shorter holding periods

**Recommendation**: Since the upstream intends bars = trade dates (business days), and min_hold=5 means 5 trading days, the consensus should be **business days**. Align all implementations to `pd.bdate_range` approach.

---

## 5. Other Upstream Changes (CalGit recent commits)

These affect `strategy.py` and `manage.py` but not the order model directly:

| Change | File | Impact |
|--------|------|--------|
| `strategy_identifier` field on `ComplexStrategySetDefinition` | `manage.py`, `strategy.py` | Needed if you use complex simulations |
| Annual returns output for complex sims | `manage.py` | Cosmetic, requires the metric fields |
| Sort s4/s6 entry signals by above_ratio | `manage.py`, `strategy.py` | Different signal ordering affects trade priority |

**Upstream strategy.py** (2,916 lines) is significantly simpler than **JimGit strategy.py** (4,462 lines). The extra ~1,546 lines cover:
- `StrategyEntryFilters` with chip metric filters (d_sma, ema, d_ema, price_score, near_delta, price_tightness)
- Extended `TradeDetail` with MFE/MAE, excursion tracking, signal_bar_open
- Complex simulation with multiple strategy sets (A/B/C with global caps)
- SIC code exclusions (6221, 6770)

These are your downstream additions and should be carried forward when merging upstream changes.

---

## Summary: Action Items

| # | Priority | What | Where |
|---|----------|------|-------|
| 1 | **HIGH** | Update `FILTER_STR` → `Top500` with `0.02%` threshold | `order_model_v2.py:113`, `order_model_v2_batch.py:201` |
| 2 | **HIGH** | Update `_get_top_symbols_for_date` default → `top_n: int = 500` | `order_model_v2.py:236` |
| 3 | **HIGH** | Update `fast_signal_scan` call → `top_n=500` | `order_model_v2.py:406` |
| 4 | **HIGH** | Remove min_hold block on SL → set `sl_activation = t1_date` | `order_model_v2.py:1001-1002` |
| 5 | **HIGH** | Place SL immediately (no min_hold deferral) | `place_tp_sl.py:268-273` |
| 6 | **MEDIUM** | Align `_bars_between` to business days everywhere | `order_engine.py:361`, `order_model_v2.py:563` |
| 7 | **MEDIUM** | Update legacy test scripts Top200→Top500 | `test_rdt*.py`, `debug_*.py` |
| 8 | **LOW** | Update JSON backtest configs to `0.02%,Top500,Pick5` | `data/*.json` (except production) |
| 9 | **LOW** | Forward-port upstream complex sim changes | `manage.py`, `strategy.py` |

---

## Appendix: Current State Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                  FILTER UNIVERSE INCONSISTENCY                    │
├────────────────────────────────────────────────────────────────┤
│                                                               │
│  Production (daily_job):     Top500 ──→  ~150 symbols          │
│  Config (production JSON):   Top500 ──→  ~150 symbols          │
│  order_model_v2.py:          Top200 ──→   ~55 symbols      ❌  │
│  order_model_v2_batch.py:    Top200 ──→   ~55 symbols      ❌  │
│  Test/Debug scripts:         Top200 ──→   ~55 symbols      ❌  │
│                                                               │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                SL MIN_HOLD INCONSISTENCY                         │
├────────────────────────────────────────────────────────────────┤
│                                                               │
│  Upstream design:  SL on T+1, no block   Exit: min_hold=5      │
│  order_engine.py:  SL on T+1, no block   Exit: min_hold=5  ✅  │
│  order_model_v2.py: SL on T+MIN_HOLD     Exit: min_hold=5  ❌  │
│  place_tp_sl.py:    SL deferred until    Exit: N/A         ❌  │
│                     min_hold satisfied                         │
│                                                               │
└────────────────────────────────────────────────────────────────┘
```
