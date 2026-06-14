# Backtest Analysis

## Overview

The stock_indicator system supports running backtests to evaluate trading strategies over historical price data. Backtests are executed through the management shell using `start_simulate` commands.

## Running Backtests

### Command Format

```bash
python -m stock_indicator.manage
(start_simulate) start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] [margin=NUMBER] DOLLAR_VOLUME_FILTER [STOP_LOSS] [SHOW_DETAILS] strategy=ID [group=1,2,...]
```

### Example Commands

```bash
# Basic backtest with default strategy
start_simulate dollar_volume>1 ema_sma_cross ema_sma_cross 1.0

# With custom start date and cash
start_simulate start=2020-01-01 starting_cash=5000 dollar_volume>1 ema_sma_cross ema_sma_cross 1.0

# Using strategy sets
start_simulate dollar_volume>0.05%,Top20,Pick7 strategy=s6 1.0
```

## Backtest Output Format

### Summary Metrics (Printed to Console)

When a backtest completes, the system prints:

```
Simulation start date: 2020-01-01
Trades: 150, Win rate: 52.33%, Mean profit %: 8.45%, Profit % Std Dev: 6.12%, Mean loss %: -5.23%, Loss % Std Dev: 3.45%, Mean holding period: 12.34 bars, Holding period Std Dev: 8.76 bars, Max concurrent positions: 3, Final balance: 14523.67, CAGR: 18.23%, Max drawdown: -12.45%
Year 2020: 15.23%, trade: 45
Year 2021: 22.45%, trade: 38
...
```

**Key Metrics Explained:**
- **Trades**: Total number of completed trades
- **Win rate**: Percentage of profitable trades (wins / total trades)
- **Mean profit %**: Average percentage gain on winning trades
- **Profit % Std Dev**: Standard deviation of profit percentages
- **Mean loss %**: Average percentage loss on losing trades (negative value)
- **Loss % Std Dev**: Standard deviation of loss percentages
- **Mean holding period**: Average number of bars held per position
- **Max concurrent positions**: Maximum simultaneous positions held
- **Final balance**: Ending portfolio value
- **CAGR**: Compound Annual Growth Rate
- **Max drawdown**: Largest peak-to-trough decline percentage

### Detailed Trade CSV

Backtest results are saved to:
- **Directory**: `logs/simulate_result/`
- **Filename**: `simulation_YYYYMMDD_HHMMSS.csv`

**CSV Columns:**
- `year` - Trade year
- `entry_date` - Entry date
- `concurrent_position_index` - Position number when trade opened
- `symbol` - Ticker symbol
- `price_concentration_score` - Price concentration metric
- `near_price_volume_ratio` - Volume near current price ratio
- `above_price_volume_ratio` - Volume above current price ratio
- `below_price_volume_ratio` - Volume below current price ratio
- `near_delta` - Near price delta metric
- `price_tightness` - Price tightness metric
- `histogram_node_count` - Volume profile histogram nodes
- `sma_angle` - Simple moving average angle in degrees
- `d_sma_angle` - Derivative of SMA angle
- `ema_angle` - Exponential moving average angle
- `d_ema_angle` - Derivative of EMA angle
- `signal_bar_open` - Open price on signal date
- `exit_date` - Exit date
- `result` - "win" or "lose"
- `percentage_change` - Profit/loss percentage
- `max_favorable_excursion_pct` - Best price during holding period
- `max_adverse_excursion_pct` - Worst price during holding period
- `max_favorable_excursion_date` - Date of best price
- `max_adverse_excursion_date` - Date of worst price
- `exit_reason` - How trade exited (signal, stop_loss, take_profit)
- `profit_per_bar` - Daily profit percentage

### Trade Details Log

Additional trade-by-trade logs are saved to:
- **Directory**: `logs/trade_detail/`
- **Filename**: `trade_details_YYYYMMDD_HHMMSS.log`

Format:
```
YYYY-MM-DD (N) SYMBOL open PRICE VOLUME_RATIO near_pct=X.XX
YYYY-MM-DD (N) SYMBOL close PRICE win 12.34% exit_signal
YYYY-MM-DD (N) SYMBOL close PRICE lose -5.67% stop_loss
```

## Available Strategies

From `data/strategy_sets.csv` and `Docs/Usage.md`:

### Entry/Buy Strategies
- `ema_sma_cross` - Basic EMA crosses above SMA
- `20_50_sma_cross` - 20-day SMA crosses 50-day SMA
- `ema_sma_cross_with_slope` - EMA/SMA cross with slope filter
- `ema_sma_cross_testing` - Advanced EMA/SMA cross with chip concentration filters
- `ema_shift_cross_with_slope` - **Breakthrough: EMA crosses shifted EMA (t-3)**
- `ftd_ema_sma_cross` - (removed in recent version)
- `ema_sma_cross_and_rsi` - EMA/SMA cross with RSI filter
- `ema_sma_cross_with_slope_and_volume` - EMA/SMA cross with volume filters

### Strategy Sets (from data/strategy_sets.csv)

| ID | Buy Strategy | Sell Strategy | Description |
|-----|--------------|----------------|-------------|
| s1 | ema_sma_cross_testing_4_-0.5_99 | ema_sma_cross_testing_5 | Near price volume filter |
| s2 | ema_sma_cross_testing_4_-0.5_99 | ema_sma_cross_testing_5 | Similar to s1 |
| s3 | ema_sma_cross_testing_40_-16.7_65 | ema_sma_cross_testing_50 | Standard slope range |
| s4 | ema_sma_cross_testing_4_-0.01_65_-10.0,10.0_0.78,1.00 | ema_sma_cross_testing_5 | Tight slope + volume filters |
| s5 | ema_sma_cross_testing_4_-99_99_0.78,1.00 | ema_sma_cross_testing_3 | Reverse volume filter on sell |
| s6 | ema_sma_cross_testing_4_0.75_10_0.0,0.3_0.0,0.78 | ema_sma_cross_testing_5 | Specific slope range |
| buy3 | ema_sma_cross_testing_3_-99_99_0.973,1.0 | ema_sma_cross_testing_3 | Tight above-price filter |
| near_close | ema_sma_cross_testing_3_-99_99_0.0,0.09_0.0,0.973 | ema_sma_cross_testing_15 | Very tight near filter |

### Important Note
**Strategy sets only use `ema_sma_cross_testing` strategies** - none use the breakthrough `ema_shift_cross_with_slope` strategy! This suggests:
1. The breakthrough strategy may be experimental
2. It may have been tried but not adopted for production
3. Performance comparison may not have been run yet

## Data Source

Backtest data is loaded from:
- **Directory**: `data/stock_data/`
- **Format**: CSV files per symbol (e.g., `AAPL.csv`)
- **Columns**: `Date,close,high,low,open,volume`

### Data Availability

Current data appears to be from 2025-2026 (very recent). Historical backtests may not work unless historical data is downloaded first using:

```bash
python -m stock_indicator.manage
(start_simulate) update_all_data_from_yf 2020-01-01 2023-12-31
```

## Missing Backtest Results

**Status**: No existing backtest CSV files found in `logs/simulate_result/`

**To generate backtests**:
1. Download historical data: `update_all_data_from_yf START_DATE END_DATE`
2. Run simulation: `start_simulate ...`
3. Results saved to `logs/simulate_result/simulation_*.csv`

## Multi-Bucket Simulations

For more advanced testing, use JSON configuration files:

**Example**: `data/multi_bucket_example.json`
```json
{
  "max_position_count": 30,
  "starting_cash": 300000,
  "start_date": "2014-01-01",
  "withdraw": 0,
  "min_hold": 5,
  "margin": 1.0,
  "confirmation_mode": null,
  "buckets": [
    {
      "label": "B1_nearPV",
      "strategy_id": "s51",
      "dollar_volume_filter": "dollar_volume>0.05%,Top50,Pick2",
      "stop_loss": 0.109,
      "take_profit": 0.084,
      "priority": 1,
      "max_positions": null
    }
  ]
}
```

Run with:
```bash
python -m stock_indicator.manage
(start_simulate) multi_bucket_simulation data/multi_bucket_example.json
```

Output: `logs/multi_bucket_simulation_result/multi_bucket_simulation_*.csv`

## Next Steps to Compare Breakthrough vs Traditional

To verify the breakthrough strategy's success rate improvement:

1. **Download historical data** (10+ years):
   ```bash
   update_all_data_from_yf 2010-01-01 2023-12-31
   ```

2. **Add shift strategy to strategy_sets.csv**:
   ```csv
   shift_test,ema_shift_cross_with_slope_35,ema_shift_cross_with_slope_35
   ```

3. **Run comparison backtests**:
   ```bash
   # Traditional
   start_simulate start=2010-01-01 dollar_volume>0.1%,Top50,Pick2 strategy=s3 1.0 false > logs/traditional.log

   # Breakthrough
   start_simulate start=2010-01-01 dollar_volume>0.1%,Top50,Pick2 ema_shift_cross_with_slope ema_shift_cross_with_slope 1.0 false > logs/breakthrough.log
   ```

4. **Compare metrics**:
   - Win rate
   - Mean profit %
   - CAGR
   - Max drawdown

## Related Files

- `src/stock_indicator/manage.py` - Backtest command handlers (lines 1759-2395)
- `src/stock_indicator/simulator.py` - Trade simulation engine
- `src/stock_indicator/strategy.py` - Signal generation logic
- `data/strategy_sets.csv` - Strategy configurations
- `logs/simulate_result/` - Backtest CSV outputs
- `logs/trade_detail/` - Trade-by-trade logs
