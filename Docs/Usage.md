# Usage

To evaluate the FTD EMA and SMA cross strategy in the management shell, call:

```
start_simulate dollar_volume>50 ftd_ema_sma_cross ftd_ema_sma_cross
```

The previous `start_ftd_ema_sma_cross` command has been removed.
Use `start_simulate` with `ftd_ema_sma_cross` for both the buying and
selling strategies instead.

## Available strategies

The `start_simulate` command accepts the following strategies:

* `ema_sma_cross`
* `ema_sma_cross_and_rsi`
* `ftd_ema_sma_cross`
* `ema_sma_cross_with_slope`
* `kalman_filtering` *(sell only)*

Not every strategy supports both buying and selling. Only the first four
strategies in the list can be used for buying. All five strategies can be used
for selling.
