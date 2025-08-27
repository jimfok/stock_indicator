# Stock Indicator

## Project Description
Stock Indicator provides a collection of Python utilities for computing common technical indicators used in stock market analysis. The goal is to make it easy to calculate metrics such as moving averages, RSI, and MACD on historical price data.

## Quick Start

### Requirements
- Python 3.10+
- Packages: `pandas`, `numpy`, `yfinance`, `matplotlib`
- Internet connection for downloading market data from providers like [Yahoo Finance](https://finance.yahoo.com) or [Alpha Vantage](https://www.alphavantage.co/)

### Installation
```bash
git clone https://github.com/yourusername/stock_indicator.git
cd stock_indicator
pip install pandas numpy yfinance matplotlib
```

### Example Usage
```python
from stock_indicator.data_loader import download_history
from stock_indicator.indicators import rsi

prices = download_history("AAPL", "2023-01-01", "2023-06-01")
prices["rsi_14"] = rsi(prices["close"], window=14)
print(prices[["close", "rsi_14"]].tail())
```

Downloaded data frames use lower-case ``snake_case`` column names. With
``yfinance`` version ``0.2.51`` and later, the ``close`` column already reflects
dividends and stock splits, so no separate adjusted closing price column is
provided. Downstream code should refer to columns using this standardized
style.

### Command Line Example
Stock Indicator also includes a command line interface for generating trade signals from historical price data.

```bash
python -m stock_indicator.cli --symbol AAPL --start 2023-01-01 --end 2023-06-01 --strategy sma --output trades.csv
```

* `--symbol` — ticker symbol of the stock to analyze.
* `--start` — start date for the price history in `YYYY-MM-DD` format.
* `--end` — end date for the price history in `YYYY-MM-DD` format.
* `--strategy` — indicator or strategy to apply, such as `sma` for simple moving average.
* `--output` — file path for saving generated trades as a CSV file.

### Management Shell

The package provides an interactive shell for updating the symbol cache and
downloading historical price data.

```bash
python -m stock_indicator.manage

(stock-indicator) update_symbols
(stock-indicator) update_data AAPL 2024-01-01 2024-02-01
(stock-indicator) update_all_data 2024-01-01 2024-02-01
(stock-indicator) exit
```

* `update_symbols` downloads the latest list of available ticker symbols.
* `update_data SYMBOL START END` saves historical data for the given symbol to
  `data/<SYMBOL>.csv`.
* `update_all_data START END` performs the download for every cached symbol.
* `find_signal DATE DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY STOP_LOSS`
  recalculates the entry and exit signals for `DATE` using the provided
  strategies instead of reading log files.

For example:

```bash
(stock-indicator) find_signal 2024-01-10 dollar_volume>1 ema_sma_cross ema_sma_cross 1.0
['AAA', 'BBB']
['CCC', 'DDD']
```

Developers can also call `daily_job.find_signal("2024-01-10", "dollar_volume>1", "ema_sma_cross", "ema_sma_cross", 1.0)` to compute
the same data from Python code. This function recalculates signals rather than
reading them from log files.

The shell can also simulate trading strategies. The `dollar_volume` filter
accepts a minimum threshold in millions, a percentage of total market volume,
and a ranking when combined with a comma. The command below evaluates
`ftd_ema_sma_cross` using only the six symbols whose 50-day average dollar
volume exceeds 10,000 million:

```bash
(stock-indicator) start_simulate starting_cash=5000 withdraw=1000 dollar_volume>10000,6th ftd_ema_sma_cross ftd_ema_sma_cross
```

Here `dollar_volume>10000,6th` first drops symbols below the threshold and then
selects the six highest-volume symbols from the remainder. The tests
`tests/test_manage.py::test_start_simulate_dollar_volume_threshold_and_rank` and
`tests/test_strategy.py::test_evaluate_combined_strategy_dollar_volume_filter_and_rank`
demonstrate this combined syntax.

To express the threshold as a percentage of total market dollar volume, use a
percent sign. For example `dollar_volume>1%` retains only symbols whose
50-day average dollar volume is greater than one percent of the combined
volume across all symbols.

## Contribution Guidelines
1. Fork the repository and create a new branch for each feature or bug fix.
2. Ensure your code passes all tests by running `pytest` before submitting.
3. Open a pull request with a clear description of your changes.

This project is released under the MIT License. By contributing, you agree to license your work under the same terms.
