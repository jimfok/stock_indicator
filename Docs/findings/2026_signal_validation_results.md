# 2026 Entry Signal Validation Results

**Date:** 2026-05-01  
**Test:** Sample scan (first 100 symbols)  
**Data:** Isolated 2025-2026 data from stock_data_2014

---

## Signal Calculation (Production buy3 Strategy)

**Strategy:** `ema_sma_cross_testing_3_-99_99_-99.0,99.0_0.973,1.0`

**Entry Conditions:**
1. **EMA3/SMA3 crossover** at T close
   - EMA3[T-1] ≤ SMA3[T-1]
   - EMA3[T] > SMA3[T]

2. **Chip concentration filter** at T close
   - `above_price_volume_ratio[T] ∈ [0.973, 1.0]`

3. **Signal timing**
   - Signal generated: T close
   - Order placement: T+1 open
   - All filters evaluated using T-day data

---

## Scan Results (100 Symbol Sample)

| Metric | Value |
|--------|-------|
| Symbols scanned | 100 |
| Total 2026 signals | 1,403 |
| Unique symbols with signals | 88 (88%) |
| Avg signals per symbol | ~16 |
| Date range | 2026-01-02 to 2026-04-29 |

---

## Sample Signals (First 20)

```
symbol  entry_date
A       2026-01-05
A       2026-01-13
A       2026-01-22
A       2026-01-28
A       2026-02-02
AA      2026-01-05
AA      2026-01-12
AA      2026-01-21
AAAU    2026-01-02
AAAU    2026-01-12
AAPL    2026-01-09
AAPL    2026-01-23
AAPL    2026-02-02
```

---

## Signal Distribution by Symbol (Top 10)

| Symbol | Signal Count |
|--------|--------------|
| A | 17 |
| AA | 15 |
| AAAU | 17 |
| AACB | 13 |
| AACG | 17 |
| AAL | 16 |
| AAME | 15 |
| AAMI | 19 |
| AAOI | 16 |
| AAON | 17 |

---

## Next Steps

1. ✅ **Signal calculation validated** - EMA3/SMA3 + chip filter working
2. ⏭️ **Apply Order Model** - Simulate T+1 order placement, gap fills
3. ⏭️ **Add exit logic** - TP/SL/exit_signal simulation
4. ⏭️ **Compare with production** - Validate against current simulator results

---

## Files Created

- `scripts/isolate_2025_2026_data.py` — Isolates recent data for fast testing
- `scripts/scan_2026_signals_validation.py` — Full scan (all symbols)
- `scripts/scan_2026_signals_sample.py` — Sample scan (100 symbols)
- `data/stock_data_2025_2026/` — Isolated 2025-2026 data (9,033 symbols)
- `data/2026_entry_signals_sample_*.csv` — Signal output for validation

---

## Notes

- **Chip calculation is slow** — ~3 minutes for 100 symbols
- **Full scan would take ~4.5 hours** for 9,033 symbols
- **Recommendation:** Optimize chip calculation or use parallel processing for full scan
