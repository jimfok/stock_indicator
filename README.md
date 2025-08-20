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

Downloaded data frames use lower-case ``snake_case`` column names. For instance,
``"Adj Close"`` is exposed as ``"adj_close"``. Downstream code should refer to
columns using this standardized style.

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

## Contribution Guidelines
1. Fork the repository and create a new branch for each feature or bug fix.
2. Ensure your code passes all tests by running `pytest` before submitting.
3. Open a pull request with a clear description of your changes.

This project is released under the MIT License. By contributing, you agree to license your work under the same terms.
