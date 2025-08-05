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
import yfinance as yf
from stock_indicator.indicators import rsi

prices = yf.download("AAPL", period="6mo")
prices["RSI_14"] = rsi(prices["Close"], window=14)
print(prices[["Close", "RSI_14"]].tail())
```

## Contribution Guidelines
1. Fork the repository and create a new branch for each feature or bug fix.
2. Ensure your code passes all tests by running `pytest` before submitting.
3. Open a pull request with a clear description of your changes.

This project is released under the MIT License. By contributing, you agree to license your work under the same terms.
