# 2026 Entry Signal Validation - COMPLETE ✓

**Date:** 2026-05-01  
**Status:** Signal calculation validated against production backtest

---

## Production Entry Signals (Validated)

**Late April 2026:**
| Date | Symbols |
|------|---------|
| 2026-04-22 | CL |
| 2026-04-24 | ABBV |
| 2026-04-28 | LLY, ABBV, MRK |

**All 4 symbols match exactly** ✓

---

## Validated Signal Equation

```python
# Entry signal at T close (signal dated T, same-day)
entry_signal[T] = EMA_CROSS_UP[T] AND ABOVE_RATIO_OK[T]

# Where:
EMA_CROSS_UP[T] = (EMA3[T-1] <= SMA3[T-1]) AND (EMA3[T] > SMA3[T])
ABOVE_RATIO_OK[T] = (0.973 <= above_price_volume_ratio[T] <= 1.0)

# Indicators:
EMA3[T] = EMA(close, span=3)[T]
SMA3[T] = SMA(close, window=3)[T]
```

**Key parameters:**
- `window_size = 3` (EMA3/SMA3 crossover)
- `above_price_volume_ratio ∈ [0.973, 1.0]` (chip concentration filter)
- **Signal timing: SAME-DAY** (crossover date = signal date)

---

## Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/isolate_2025_2026_data.py` | Isolate 2025-2026 data (9,033 symbols) |
| `scripts/validate_final.py` | Validate against production signals |
| `scripts/scan_2026_sample500.py` | Scan 500 symbols (fast) |
| `scripts/scan_2026_signals_corrected.py` | Full scan (slow, ~4.5 hours) |

---

## Sample Scan Results (500 Symbols)

| Metric | Value |
|--------|-------|
| Symbols scanned | 500 |
| Total 2026 signals | 799 |
| Unique symbols | 253 (50.6%) |
| Avg signals/symbol | ~3.2 |

**By month:**
- 2026-01: 136 signals
- 2026-02: 222 signals
- 2026-03: 324 signals
- 2026-04: 117 signals

---

## Signal Calculation Code

```python
from stock_indicator.chip_filter import calculate_chip_concentration_metrics

def calculate_buy3_signals(df: pandas.DataFrame) -> pandas.DataFrame:
    """Calculate entry signals with SAME-DAY timing (production validated)."""
    df = df.copy()
    
    # Calculate chip concentration metrics
    near_ratios = []
    above_ratios = []
    for row_index in range(len(df)):
        chip_metrics = calculate_chip_concentration_metrics(
            df.iloc[: row_index + 1],
            lookback_window_size=60,
            include_volume_profile=False,
        )
        near_ratios.append(chip_metrics["near_price_volume_ratio"])
        above_ratios.append(chip_metrics["above_price_volume_ratio"])
    
    df["near_price_volume_ratio"] = pandas.Series(near_ratios, index=df.index)
    df["above_price_volume_ratio"] = pandas.Series(above_ratios, index=df.index)
    
    # Calculate EMA3/SMA3
    _close_r3 = df["close"].round(3)
    df["ema_value"] = pandas.Series(_close_r3).ewm(span=3, adjust=False).mean()
    df["sma_value"] = pandas.Series(_close_r3).rolling(3).mean()
    df["ema_previous"] = df["ema_value"].shift(1)
    df["sma_previous"] = df["sma_value"].shift(1)
    
    # Raw crossover (same day)
    df["ema_cross_up"] = (
        (df["ema_previous"] <= df["sma_previous"]) &
        (df["ema_value"] > df["sma_value"])
    )
    
    # Apply chip filter on same day
    df["above_ratio_ok"] = (
        df["above_price_volume_ratio"].ge(0.973) &
        df["above_price_volume_ratio"].le(1.0)
    ).fillna(False)
    
    # Entry signal = crossover AND chip filter (both at T close, signal dated T)
    df["entry_signal"] = df["ema_cross_up"] & df["above_ratio_ok"]
    
    return df
```

---

## Next Steps

1. ✅ **Signal calculation validated** - Matches production exactly
2. ⏭️ **Run full scan** - All 9,033 symbols (overnight, ~4.5 hours)
3. ⏭️ **Apply Order Model** - T+1 order placement, gap fill simulation
4. ⏭️ **Add exit logic** - TP/SL/exit_signal simulation
5. ⏭️ **Compare with production** - Full backtest validation

---

## Performance Notes

- **Chip concentration calculation is slow** - ~3 minutes per 100 symbols
- **Full scan estimate:** ~4.5 hours for 9,033 symbols
- **Recommendation:** Run overnight or optimize with batch processing

---

## Files

- `data/stock_data_2025_2026/` - Isolated 2025-2026 data (9,033 symbols)
- `data/2026_entry_signals_sample500_*.csv` - Sample scan results
- `docs/findings/2026_signal_validation_complete.md` - This document

---

**Generated:** 2026-05-01  
**Validated by:** Hermes Agent
