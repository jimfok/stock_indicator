# Signal Calculation: My Fast Scan vs Production

## Critical Differences Found

### My Fast Scan Script (INCORRECT) ❌

**File:** `scripts/scan_2026_signals_fast.py`

```python
# Simplified EMA/SMA crossover
SMA10 crosses above SMA50
EMA12 > SMA50  # confirmation
```

**Problems:**
1. Wrong windows: Used SMA10/SMA50/EMA12 instead of EMA3/SMA3
2. Missing chip concentration filter (above_price_volume_ratio)
3. Missing proper signal shifting (T close → T+1 order)
4. Not using production `attach_ema_sma_cross_testing_signals()`

---

### Production Code (CORRECT) ✅

**Source:** `strategy.py` + `strategy_sets.csv` + `manage.py`

**buy3 strategy identifier:**
```
ema_sma_cross_testing_3_-99_99_-99.0,99.0_0.973,1.0
```

**Parsed parameters:**
- `base_name`: ema_sma_cross_testing
- `window_size`: 3 (EMA3/SMA3, not EMA12/SMA50!)
- `angle_range`: (-99, 99) → effectively no filter
- `near_range`: (-99.0, 99.0) → no filter
- `above_range`: (0.973, 1.0) → **THE CHIP FILTER**

**CSV column filters for buy3:** All None (no additional filters)
- d_sma_range: None
- ema_range: None
- d_ema_range: None
- near_delta_range: None
- price_tightness_range: None
- sma_150_angle_min: None
- price_score_min/max: None

---

## Production Entry Signal Logic

```python
# Step 1: Calculate chip concentration metrics
for each bar T:
    chip_metrics = calculate_chip_concentration_metrics(df[:T+1], lookback=60)
    near_price_volume_ratio[T] = chip_metrics["near_price_volume_ratio"]
    above_price_volume_ratio[T] = chip_metrics["above_price_volume_ratio"]

# Step 2: Calculate EMA3/SMA3 crossover
ema3[T] = EMA(close, 3)
sma3[T] = SMA(close, 3)

# Step 3: Generate raw crossover signal (at T close)
raw_crossover[T] = (ema3[T-1] <= sma3[T-1]) and (ema3[T] > sma3[T])

# Step 4: Apply filters and shift by 1 day (signal at T, decision at T+1)
entry_signal[T+1] = (
    raw_crossover[T] and  # Crossover happened at T close
    (0.973 <= above_price_volume_ratio[T] <= 1.0)  # Chip filter at T
)
```

**Key timing:**
- Signal generated at **T close** (after market closes)
- Order placed at **T+1 open** (next day's open)
- All filters evaluated using **T-day data** (known at T close)

---

## Production Exit Signal Logic (buy3 sell strategy)

**Sell strategy:** `ema_sma_cross_testing_3_-0.01_65_-10.0,10.0_0.78,1.00`

**Parsed:**
- `window_size`: 3
- `angle_range`: (-0.01, 65) → SMA angle must be positive (uptrend confirmation)
- `near_range`: (-10.0, 10.0) → no effective filter
- `above_range`: (0.78, 1.00) → chip filter (looser than entry)

**Exit conditions:**
1. EMA3 crosses below SMA3 (downtrend signal)
2. SMA angle > -0.01° (not in strong downtrend)
3. above_price_volume_ratio[T] ∈ [0.78, 1.0]

---

## New Scripts Created

### 1. `scripts/backtest_buy3_2014.py`
- **Data:** stock_data_2014 (2014-2026)
- **Config:** multi_bucket_buy3_production.json
- **Purpose:** Scan all symbols for production buy3 signals

### 2. `scripts/backtest_buy3_1989.py`
- **Data:** stock_data_1989 (1989-2026)
- **Config:** Same logic, different data source
- **Purpose:** Long-term backtest with extended history

**Both scripts:**
- Use correct `attach_ema_sma_cross_testing_signals()` with buy3 parameters
- Calculate chip concentration metrics properly
- Output CSV with symbol + entry_date for validation

---

## Next Steps

1. **Run signal scan** to get all entry signals with correct calculation
2. **Validate entry dates** against production backtest results
3. **Apply Order Model** (T+1 order placement, gap fill logic)
4. **Compare results** with current simulator output

---

## Files to Review

- `src/stock_indicator/strategy.py` — `attach_ema_sma_cross_testing_signals()` (line 2130)
- `src/stock_indicator/manage.py` — Signal processing in `_generate_strategy_evaluation_artifacts()` (line 3040+)
- `data/strategy_sets.csv` — buy3 strategy definition (line 21)
- `data/multi_bucket_buy3_production.json` — Production config

---

**Generated:** 2026-05-01
**Author:** Hermes Agent
