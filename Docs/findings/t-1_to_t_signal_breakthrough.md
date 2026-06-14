# T-1 to T Signal Breakthrough

## Overview

The inventor's breakthrough strategy dramatically increased success rate by introducing a novel signal generation mechanism that compares **current momentum (t)** against **lagged momentum (t-3)** through exponential moving averages (EMAs).

## The Core Innovation

### Traditional Approach (Before Breakthrough)

**EMA/SMA Cross Strategy** (`attach_ema_sma_cross_with_slope_signals`):
- Compares current EMA (based on recent prices) against current SMA
- Signal: `EMA[t] crosses above SMA[t]`
- Problem: Both indicators use recent price data, leading to false signals during volatility

### Breakthrough Approach

**EMA Shift Cross with Slope** (`attach_ema_shift_cross_with_slope_signals`):
- Compares current EMA against an EMA calculated from **3-day-lagged prices**
- Signal: `EMA[t] crosses above shifted_EMA[t]`
- The shifted EMA acts as a stabilized momentum baseline

## Technical Implementation

### Location

**File**: `src/stock_indicator/strategy.py`
**Function**: `attach_ema_shift_cross_with_slope_signals` (lines 2078-2174)

### Key Code Snippet

```python
# Line 2132: The breakthrough - shift close prices back by 3 days
price_data_frame["shifted_close"] = price_data_frame["close"].round(3).shift(3)

# Line 2133-2135: Calculate EMA on shifted prices
price_data_frame["shifted_ema_value"] = ema(
    price_data_frame["shifted_close"], window_size
)

# Lines 2137-2138: Track previous values for cross detection
price_data_frame["ema_previous"] = price_data_frame["ema_value"].shift(1)
price_data_frame["shifted_ema_previous"] = price_data_frame["shifted_ema_value"].shift(1)

# Lines 2140-2147: Detect crosses between EMA and shifted EMA
crosses_up = (
    (price_data_frame["ema_previous"] <= price_data_frame["shifted_ema_previous"])
    & (price_data_frame["ema_value"] > price_data_frame["shifted_ema_value"])
)
crosses_down = (
    (price_data_frame["ema_previous"] >= price_data_frame["shifted_ema_previous"])
    & (price_data_frame["ema_value"] < price_data_frame["shifted_ema_value"])
)

# Lines 2149-2154: Calculate the shifted EMA angle (momentum of lagged prices)
relative_change = (
    price_data_frame["shifted_ema_value"] - price_data_frame["shifted_ema_previous"]
) / price_data_frame["shifted_ema_previous"]
price_data_frame["shifted_ema_angle"] = numpy.degrees(
    numpy.arctan(relative_change)
)

# Lines 2156-2164: Final entry signal - cross AND angle filter
base_entry = crosses_up.shift(1, fill_value=False)
price_data_frame["ema_shift_cross_with_slope_entry_signal"] = (
    base_entry
    & (shifted_ema_angle_previous >= angle_lower_bound)
    & (shifted_ema_angle_previous <= angle_upper_bound)
)
```

## Why This Increases Success Rate

### 1. Momentum Confirmation

- **EMA[t]**: Fast-responding to current price movements
- **Shifted_EMA[t]**: Based on prices from 3 days ago, representing lagged momentum
- **Cross**: When EMA[t] > shifted_EMA[t], it means **current momentum has overtaken momentum from 3 days ago**

### 2. Reduced False Positives

Traditional EMA/SMA crosses often trigger during temporary spikes because both indicators react to the same price data. The shifted EMA:
- Acts as a **stabilized baseline** (3-day smoothing)
- Filters out short-term noise
- Only fires when momentum is **sustained** over a multi-day period

### 3. Angle Filter Validation

The strategy adds an additional guardrail: the shifted EMA must be trending within a specific angle range (line 2160-2163):
- Ensures the lagged momentum is in a favorable state
- Prevents entries when the shifted EMA is declining too steeply or is flat
- Defaults to `DEFAULT_SHIFTED_EMA_ANGLE_RANGE` (values in code)

## Signal Timing

### T-1 to T Pattern

1. **Day T**: Close price → Calculate EMA[t] and shifted_EMA[t]
2. **Day T+1**: Signal is available for trading at market open
3. **Execution**: Trade placed at T+1 open based on T's close

The `.shift(1, fill_value=False)` on line 2156 ensures that:
- Crosses detected on day T become available as entry signals on day T+1
- This prevents look-ahead bias (you can't trade on today's close before market closes)

## Strategy Parameters

### Default Values

**For EMA Shift Cross with Slope** (breakthrough strategy):
- **Window Size**: 35 periods (EMA calculation window)
- **Shift Amount**: 3 days (hardcoded in line 2132)
- **Angle Range**: `DEFAULT_SHIFTED_EMA_ANGLE_RANGE = (-45°, 45°)`
  - Calculated from tangent bounds: `atan(-1.0) = -45°` and `atan(1.0) = 45°`
  - Defined in lines 40-43 of `strategy.py`

**For Comparison - EMA/SMA Cross with Slope** (traditional):
- **SMA Angle Range**: `DEFAULT_SMA_ANGLE_RANGE = (-16.7°, 65.0°)`
  - Calculated from tangent bounds: `atan(-0.3)` and `atan(2.14)`
  - Defined in lines 35-38 of `strategy.py`

**Note**: The shifted EMA strategy has a tighter angle filter (-45° to 45°) compared to traditional SMA (-16.7° to 65.0°), suggesting more selective entry criteria.

### Usage

In strategy sets (`data/strategy_sets.csv`):
```csv
strategy_id,buy,sell
s1,ema_shift_cross_with_slope_35,ema_shift_cross_with_slope_35
```

Custom parameters:
- `ema_shift_cross_with_slope_40_-5.0_50.0` → window=40, angle_range=(-5.0°, 50.0°)
- `ema_shift_cross_with_slope_-5.7_50.2` → window=35 (default), angle_range=(-5.7°, 50.2°)

## Historical Context

This strategy appears to be the result of experimentation documented in the codebase:
- Comment line 2182: "## Removed deprecated strategy: kalman_filtering"
- Comment line 2179: "## Removed deprecated strategy: ema_sma_double_cross"
- The shift strategy (`ema_shift_cross_with_slope`) remains active (line 2190)

## Further Investigation Needed

- What is `DEFAULT_SHIFTED_EMA_ANGLE_RANGE`? Search for this constant
- Why 3 days specifically? Could this be parameterized?
- Performance comparison between `ema_shift_cross_with_slope` vs `ema_sma_cross_with_slope`
- Backtest results showing success rate improvement

## Related Files

- `src/stock_indicator/strategy.py` - Main signal generation functions
- `data/strategy_sets.csv` - Strategy configuration
- `tests/test_strategy.py` - Unit tests for signal logic

## Next Steps

1. Locate `DEFAULT_SHIFTED_EMA_ANGLE_RANGE` definition
2. Review backtest results comparing shift vs non-shift strategies
3. Examine why 3-day shift was chosen (commentary or commit history)
4. Test alternative shift values (2, 4, 5 days) to optimize further
