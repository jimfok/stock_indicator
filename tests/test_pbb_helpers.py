import pandas as pd
import numpy as np
from stock_indicator import indicators


def test_get_stock_data(monkeypatch):
    sample = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.5, 1.5],
            "Close": [1.0, 2.0],
            "Adj Close": [1.0, 2.0],
            "Volume": [100, 200],
        },
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    def fake_download(symbol, start, end, interval):
        return sample

    monkeypatch.setattr(indicators.yf, "download", fake_download)
    df = indicators._get_stock_data("TEST", "1d")
    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert df["Close"].iloc[0] == 1.0
    assert df.index.tolist() == [0, 1]


def test_moving_average_checks():
    df = pd.DataFrame({
        "Close": [1, 2, 3, 2, 5],
        "Open": [1, 1, 1, 1, 1],
        "Volume": [10, 20, 30, 40, 50]
    })
    result = indicators._moving_average_checks(df)
    expected_ema = df["Close"].ewm(span=50, adjust=False).mean()
    expected_high = df["Close"].rolling(window=200, min_periods=1).max()
    assert result["EMA_50"].equals(expected_ema)
    assert result["HIGHESTCLOSE_200"].equals(expected_high)
    assert result["MA_CHECK"].equals(df["Close"] >= expected_ema)
    assert result["HIGHESTCLOSE_200_CHECK"].equals(df["Close"] >= expected_high * 0.8)


def test_rsi_check():
    df = pd.DataFrame({"Close": [1, 2, 1, 3, 4]})
    result = indicators._rsi_check(df)
    expected_rsi = pd.Series(indicators.rsi(df["Close"].tolist(), period=6))
    assert np.allclose(result["RSI6"], expected_rsi, equal_nan=True)
    expected_check = result["RSI6"] > result["RSI6"].shift(1)
    assert result["RSI6_CHECK"].equals(expected_check)


def test_volume_check():
    df = pd.DataFrame({"Volume": [100, 200, 150, 300, 250]})
    result = indicators._volume_check(df)
    expected_ma = df["Volume"].ewm(span=50, adjust=False).mean()
    expected_high = df["Volume"].rolling(window=10, min_periods=1).max()
    expected_check = (df["Volume"] <= expected_high * 0.5) & (df["Volume"] <= expected_ma)
    assert result["MA_VOL50"].equals(expected_ma)
    assert result["HIGHESTVOL_10"].equals(expected_high)
    assert result["VOL_CHECK"].equals(expected_check)
