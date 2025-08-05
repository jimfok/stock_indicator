import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import datetime
from typing import Sequence, Tuple, Union

from .utils import load_stock_history

"""Collection of technical indicator calculations and screeners."""

__all__ = [
        "ema",
        "sma",
        "rsi",
        "pbb",
        "ftd",
        "K1",
        "buyRating",
        "vol_up",
        "mm_stage2",
]

# yfinance dataFrame columns:
# Date Open High Low Close Adj Close Volume

# helper functions
def futu_ema(x, length, _y):
        return round((x * 2 + _y * (length - 1)) / (length + 1), 7)


def futu_sma(x, weight, length, _y):
        #SMA(X,N,M), such as Y=(X*M + Y'*(N-M))/N
        return round((x * weight + _y * (length - weight)) / length, 7)


def linear_regression_trend(ma_values: list) -> list:
	"""
	Calculate the slope and R^2 value for the given 200-day MA values over 20 days.

	Args:
	- ma_values (list): A list containing the 200-day MA values of the last 20 days.

	Returns:
	- A list containing two values: [slope, R^2 value].
	"""
	if len(ma_values) != 20:
		raise ValueError("The input list should contain exactly 20 values.")
	
	# Define the x values as day indices
	x = np.array(range(1, 21))
	y = np.array(ma_values)
	
	# Fit the linear regression model
	coefficients, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
	slope, intercept = coefficients
	
	# Calculate the R^2 value
	total_variance = np.var(y) * len(y)
	if total_variance == 0:
		r_squared = 1
	else:
		r_squared = 1 - (residuals[0] / total_variance)

	return [slope, r_squared]


def ema(prices: Sequence[float], period: int = 12):
	"""Compute the Exponential Moving Average (EMA) for a price series."""
	series = pd.Series(prices, dtype="float64")
	return series.ewm(span=period, adjust=False).mean().to_list()


def sma(prices: Sequence[float], period: int = 14):
	"""Compute the Simple Moving Average (SMA) for a price series."""
	series = pd.Series(prices, dtype="float64")
	return series.rolling(window=period, min_periods=1).mean().to_list()


def rsi(prices: Sequence[float], period: int = 14):
	"""Compute the Relative Strength Index (RSI)."""
	series = pd.Series(prices, dtype="float64")
	delta = series.diff()
	up = delta.clip(lower=0)
	down = -delta.clip(upper=0)
	roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
	roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
	rs = roll_up / roll_down
	rsi_series = 100 - (100 / (1 + rs))
	return rsi_series.fillna(0).to_list()




def _moving_average_checks(df: pd.DataFrame) -> pd.DataFrame:
	"""Add moving average related columns and checks."""
	df = df.copy()
	df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
	df["MA_CHECK"] = df["Close"] >= df["EMA_50"]
	df["HIGHESTCLOSE_200"] = df["Close"].rolling(window=200, min_periods=1).max()
	df["HIGHESTCLOSE_200_CHECK"] = df["Close"] >= df["HIGHESTCLOSE_200"] * 0.8
	return df


def _rsi_check(df: pd.DataFrame) -> pd.DataFrame:
	"""Add RSI6 column and check if RSI is rising."""
	df = df.copy()
	period = 6
	delta = df["Close"].diff()
	up = delta.clip(lower=0)
	down = -delta.clip(upper=0)
	roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
	roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
	rs = roll_up / roll_down
	df["RSI6"] = 100 - (100 / (1 + rs))
	df["RSI6_CHECK"] = df["RSI6"] > df["RSI6"].shift(1)
	return df


def _volume_check(df: pd.DataFrame) -> pd.DataFrame:
	"""Add volume related moving averages and checks."""
	df = df.copy()
	df["MA_VOL50"] = df["Volume"].ewm(span=50, adjust=False).mean()
	df["HIGHESTVOL_10"] = df["Volume"].rolling(window=10, min_periods=1).max()
	df["VOL_CHECK"] = (
		(df["Volume"] <= df["HIGHESTVOL_10"] * 0.5)
		& (df["Volume"] <= df["MA_VOL50"])
	)
	return df


def pbb(
        symbol: str,
        buy_mark_day: int,
        price_above: float,
        volumn_above: float,
        INTERVAL: str,
        debug: bool,
) -> Union[pd.DataFrame, bool]:
        """Evaluate the PBB indicator for a given symbol.

        Args:
        - symbol: Stock ticker to analyse.
        - buy_mark_day: Number of recent days to look back for a buy signal.
        - price_above: Minimum allowed closing price for the latest day.
        - volumn_above: Minimum allowed trading volume for the latest day.
        - INTERVAL: Data interval passed to ``load_stock_history``.
        - debug: When ``True`` return the full DataFrame of calculated values
          instead of a boolean result.

        Returns:
        - A DataFrame with all computed columns when ``debug`` is ``True``.
        - Otherwise, a boolean indicating whether the indicator signalled a
          buy within the last ``buy_mark_day`` days.

        Side Effects:
        - Downloads data via :func:`load_stock_history`.
        - Prints progress and execution time to stdout.
        """
        df_stock = load_stock_history(symbol, INTERVAL)

        if df_stock.empty:
                return False
        elif df_stock["Close"].iloc[-1] < price_above:
                return False
        elif df_stock["Volume"].iloc[-1] < volumn_above:
                return False
        elif len(df_stock.index) < 10:
                return False
        else:
                start_time = time.time()

                df_stock = _moving_average_checks(df_stock)
                df_stock = _rsi_check(df_stock)
                df_stock = _volume_check(df_stock)

                df_stock["UP_CHECK"] = df_stock["Close"] > df_stock["Close"].shift(1).fillna(df_stock["Close"])
                df_stock["UP20_CHECK"] = df_stock["Close"] > df_stock["Close"].shift(20).fillna(df_stock["Close"])
                df_stock["TODAY_RAISE_CHECK"] = df_stock["Close"] > df_stock["Open"]
                df_stock["YESTERDAY_DROP_CHECK"] = (
                        df_stock["Open"].shift(1).fillna(df_stock["Open"]) >
                        df_stock["Close"].shift(1).fillna(df_stock["Close"])
                )

                df_stock["STATE"] = (
                        df_stock["MA_CHECK"]
                        & df_stock["HIGHESTCLOSE_200_CHECK"]
                        & df_stock["RSI6_CHECK"]
                        & df_stock["VOL_CHECK"]
                        & df_stock["UP_CHECK"]
                        & df_stock["UP20_CHECK"]
                        & df_stock["TODAY_RAISE_CHECK"]
                        & df_stock["YESTERDAY_DROP_CHECK"]
                )

                end_time = time.time()
                print(symbol, "DONE", end_time - start_time)

                if debug:
                        return df_stock.copy()
                else:
                        return df_stock["STATE"].tail(buy_mark_day).any()

def ftd(
        symbol: str,
        buy_mark_day: int,
        price_above: float,
        volumn_above: float,
        INTERVAL: str,
        debug: bool,
) -> Tuple[Union[pd.DataFrame, bool], float]:
        """Check for the FTD (failure to deliver) indicator and rating.

        Args:
        - symbol: Stock ticker to analyse.
        - buy_mark_day: Number of recent days to look back for a signal.
        - price_above: Minimum allowed closing price for the latest day.
        - volumn_above: Minimum allowed trading volume for the latest day.
        - INTERVAL: Data interval passed to ``load_stock_history``.
        - debug: When ``True`` return the full DataFrame of calculated values.

        Returns:
        - Tuple of ``(data, rating)`` where ``data`` is either the full
          DataFrame (when ``debug`` is ``True``) or a boolean indicating if a
          signal was found, and ``rating`` is the computed buy rating.

        Side Effects:
        - Downloads data via :func:`load_stock_history`.
        - Prints progress and execution time to stdout.
        """
        df_stock = load_stock_history(symbol, INTERVAL)

	if df_stock.empty:
		return False, 0.0
	elif df_stock['Close'].iloc[-1] < price_above:		# check price higher than requirement
		return False, 0.0
	elif df_stock['Volume'].iloc[-1] < volumn_above:	# check price higher than requirement
		return False, 0.0
	elif len(df_stock.index) < 10:						# new stock within 10 trading day, SKIP
		return False, 0.0
	else:
		start_time = time.time()
		MAX_INDEX = len(df_stock.index)

		LIST_LOW = df_stock['Low'].copy().to_list()
		LIST_HIGH = df_stock['High'].copy().to_list()
		LIST_CLOSE = df_stock['Close'].copy().to_list()
		LIST_OPEN = df_stock['Open'].copy().to_list()
		LIST_VOL = df_stock['Volume'].copy().to_list()

		# MA_CHECK := REF(C,3) < REF(MA(C,50),3)
		LIST_MA_50 = []
		_last_futu_SMA_50 = LIST_CLOSE[0]
		# BOTTOM_CHECK := LLV(LOW, 23) = REF(LOW,3)
		LLV_23 = []
		# VOL_CHECK := THIS_VOL > REF(THIS_VOL,3)
		LIST_THIS_VOL = []
		# NEW volume MA check to filter volume low stocks
		LIST_MA_VOL_50 = []
		_last_futu_SMA_VOL_50 = LIST_VOL[0]
		# FTD rating
		LIST_BUYPOWER = []
		LIST_SELLPOWER = []
		LIST_RATING = []

		rolling_buy_power = 0.0
		rolling_sell_power = 0.0

		for i in range(0, MAX_INDEX):
			_futu_SMA_50 = futu_ema(LIST_CLOSE[i], 50, _last_futu_SMA_50)
			_last_futu_SMA_50 = _futu_SMA_50
			LIST_MA_50.append(_futu_SMA_50)
			
			_low = LIST_LOW[i]
			if i > 0 and i < 22:
				_temp_low = LLV_23[-1]
				if _temp_low < _low:
					_low = _temp_low
			elif i != 0:
				extra_one = i - 22 - 1
				if LIST_LOW[extra_one] > LLV_23[-1]:
					_low = LLV_23[-1]
				else:
					for j in range(-22, 0):
						_temp_low = LIST_LOW[0] if i+j<0 else LIST_LOW[i+j]
						if _temp_low < _low:
							_low = _temp_low
			LLV_23.append(_low)
			
			_vol_1 = LIST_VOL[i]
			_vol_2 = LIST_VOL[0] if i-1<0 else LIST_VOL[i-1]
			_vol_3 = LIST_VOL[0] if i-2<0 else LIST_VOL[i-2]
			_vol_4 = LIST_VOL[0] if i-3<0 else LIST_VOL[i-3]
			LIST_THIS_VOL.append(_vol_1+_vol_2+_vol_3+_vol_4)
			_futu_SMA_vol_50 = futu_ema(LIST_VOL[i], 50, _last_futu_SMA_VOL_50)
			_last_futu_SMA_VOL_50 = _futu_SMA_vol_50
			LIST_MA_VOL_50.append(_futu_SMA_vol_50)

			#    vol buy power > sell power
			#    BUY_POWER:= C - L;
			#    SELL_POWER:= H - C;
			#    TOTAL_POWER:= BUY_POWER + SELL_POWER;
			#    BUY_POWER_RATIO:= BUY_POWER / TOTAL_POWER;
			#    SELL_POWER_RATIO:= SELL_POWER / TOTAL_POWER;
			buy = LIST_CLOSE[i] - LIST_LOW[i]
			sell = LIST_HIGH[i] - LIST_CLOSE[i]
			total = buy + sell
			buy_power = 0 if total == 0 else buy / total
			sell_power = 0 if total == 0 else sell / total
			LIST_BUYPOWER.append(buy_power)
			LIST_SELLPOWER.append(sell_power)
			# rating: TOTAL_4DAY_BUY_SELL_RATIO := TOTAL_4DAY_BUY_POWER / TOTAL_4DAY_SELL_POWER

			if i < 4:
				rolling_buy_power += buy_power
				rolling_sell_power += sell_power
			else:
				rolling_buy_power += buy_power - LIST_BUYPOWER[i-4]
				rolling_sell_power += sell_power - LIST_SELLPOWER[i-4]

			# Avoid division by zero
			total_4day_buy_sell_ratio = (rolling_buy_power / rolling_sell_power) if rolling_sell_power != 0 else 99
			LIST_RATING.append(total_4day_buy_sell_ratio)

		# LOW1 := LOW
		# LOW2 := REF(LOW,1)
		# LOW3 := REF(LOW,2)
		# LOW4 := REF(LOW,3)
		# LOW_CHECK := LOW1> LOW2 AND LOW2 > LOW3 AND LOW3 > LOW4
		LIST_MA_CHECK = []
		LIST_BOTTOM_CHECK = []
		LIST_LOW_CHECK = []
		LIST_VOL_CHECK = []
		LIST_MA_VOL_CHECK = []
		for i in range(0, MAX_INDEX):
			_ma_a = LIST_CLOSE[0] if i-3<0 else LIST_CLOSE[i-3]
			_ma_b = LIST_MA_50[0] if i-3<0 else LIST_MA_50[i-3]
			LIST_MA_CHECK.append(_ma_a < _ma_b)
			_bottom_a = LLV_23[i]
			_bottom_b = LIST_LOW[0] if i-3<0 else LIST_LOW[i-3]
			LIST_BOTTOM_CHECK.append(_bottom_a == _bottom_b)
			_low_1 = LIST_LOW[i]
			_low_2 = LIST_LOW[0] if i-1<0 else LIST_LOW[i-1]
			_low_3 = LIST_LOW[0] if i-2<0 else LIST_LOW[i-2]
			_low_4 = LIST_LOW[0] if i-3<0 else LIST_LOW[i-3]
			_low_a = _low_1 > _low_2
			_low_b = _low_2 > _low_3
			_low_c = _low_3 > _low_4
			LIST_LOW_CHECK.append(_low_a and _low_b and _low_c)
			_vol_a = LIST_THIS_VOL[i]
			_vol_b = LIST_THIS_VOL[0] if i-3<0 else LIST_THIS_VOL[i-3]
			LIST_VOL_CHECK.append(_vol_a > _vol_b)
			_vol_higher_than_ma = False
			for j in range(-6, 1):
				_temp_vol = LIST_VOL[0] if i+j<0 else LIST_VOL[i+j]
				_temp_ma_vol = LIST_MA_VOL_50[0] if i+j<0 else LIST_MA_VOL_50[i+j]
				if not _vol_higher_than_ma:
					_vol_higher_than_ma = _temp_vol > _temp_ma_vol
			LIST_MA_VOL_CHECK.append(_vol_higher_than_ma)

		# FTD_CHECK := MA_CHECK AND BOTTOM_CHECK AND LOW_CHECK AND VOL_CHECK
		LIST_STATE = []
		for i in range(0, MAX_INDEX):
			LIST_STATE.append(LIST_MA_CHECK[i] and LIST_BOTTOM_CHECK[i] and LIST_LOW_CHECK[i] and LIST_VOL_CHECK[i] and LIST_MA_VOL_CHECK[i])

		end_time = time.time()
		print(symbol, 'DONE', end_time - start_time)

		output = False
		rating = 0.0

		if debug:
			# return the full df_stock
			df_stock.loc[:, 'MA_CHECK'] = LIST_MA_CHECK
			df_stock.loc[:, 'BOTTOM_CHECK'] = LIST_BOTTOM_CHECK
			df_stock.loc[:, 'LOW_CHECK'] = LIST_LOW_CHECK
			df_stock.loc[:, 'VOL_CHECK'] = LIST_VOL_CHECK
			df_stock.loc[:, 'state'] = LIST_STATE
			output = df_stock.copy()
		else:
			# return a boolean for TRUE
			for i in range(-1*buy_mark_day, 0):
				if LIST_STATE[i]:
					output = True
					rating = round(LIST_RATING[i], 2)

		return output, rating

def K1(
        symbol: str,
        buy_mark_day: int,
        price_above: float,
        volumn_above: float,
        INTERVAL: str,
        debug: bool,
) -> Union[pd.DataFrame, bool]:
        """Evaluate the K1 indicator for a given symbol.

        Args:
        - symbol: Stock ticker to analyse.
        - buy_mark_day: Number of recent days to look back for buy/sell marks.
        - price_above: Minimum allowed closing price for the latest day.
        - volumn_above: Minimum allowed trading volume for the latest day.
        - INTERVAL: Data interval passed to ``load_stock_history``.
        - debug: When ``True`` return the full DataFrame of calculated values.

        Returns:
        - DataFrame with all computed columns when ``debug`` is ``True``.
        - Otherwise, boolean indicating whether the most recent buy signal
          occurred after the most recent sell signal within the last
          ``buy_mark_day`` days.

        Side Effects:
        - Downloads data via :func:`load_stock_history`.
        - Prints progress and execution time to stdout.
        """
        df_stock = load_stock_history(symbol, INTERVAL)

	if df_stock.empty:
		return False
	elif df_stock['Close'].iloc[-1] < price_above:		# check price higher than requirement
		return False
	elif df_stock['Volume'].iloc[-1] < volumn_above:	# check price higher than requirement
		return False
	elif len(df_stock.index) < 10:						# new stock within 10 trading day, SKIP
		return False
	else:
		start_time = time.time()
		MAX_INDEX = len(df_stock.index)

		df_stock.loc[:, 'HLC'] = round((df_stock['High'] + df_stock['Low'] + df_stock['Close']) / 3, 3)
		LIST_HIGH = df_stock['High'].copy().to_list()
		LIST_LOW = df_stock['Low'].copy().to_list()
		LIST_CLOSE = df_stock['Close'].copy().to_list()
		LIST_HLC = df_stock['HLC'].copy().to_list()

		LIST_HD = []
		LIST_LD = []
		for i in range(0, MAX_INDEX):
			LIST_HD.append(LIST_HIGH[i] - (LIST_HIGH[i] if i==0 else LIST_HIGH[i-1]))
			LIST_LD.append((LIST_LOW[i] if i==0 else LIST_LOW[i-1]) - LIST_LOW[i])

		df_stock.loc[:, 'cci_mean'] = 0
		df_stock.loc[:, 'cci_avedev'] = 0
		df_stock.loc[:, 'cci12'] = 0

		df_stock.loc[:, 'DMP'] = 0
		df_stock.loc[:, 'DMM'] = 0
		df_stock.loc[:, 'TR1'] = 0
		df_stock.loc[:, 'PDI_ema8'] = 0
		df_stock.loc[:, 'MDI_ema8'] = 0

		# init before for
		LIST_DIF = []							# MACD
		LIST_DEA = []							# MACD
		LIST_MACD = []							# MACD
		LIST_D_DIF = [0]						# MACD
		LIST_D_DEA = [0]						# MACD
		LIST_RSI12 = []							# RSI
		LIST_D_RSI12 = [0]						# RSI
		LIST_MA4_D_RSI12 = []					# RSI
		LIST_BUY_RATING = []					# KUSHI BUY_RATING
		MAX_cci_mean = 0.0						# cci mean MAX for normalize
		MIN_cci_mean = 0.0						# cci mean MIN for normalize
		MAX_RSI_BR = 0.0						# BuyRating MAX for normalize
		MIN_RSI_BR = 0.0						# BuyRating MIN for normalize

		# 1st for loop
		_last_futu_EMA_12 = LIST_CLOSE[0]		# MACD
		_last_futu_EMA_26 = LIST_CLOSE[0]		# MACD
		_last_ma4_d_rsi12 = LIST_D_RSI12[0]		# RSI
		for i in range(0, MAX_INDEX):
			# MACD
			_futu_EMA_12 = futu_ema(LIST_CLOSE[i], 12, _last_futu_EMA_12)
			_futu_EMA_26 = futu_ema(LIST_CLOSE[i], 26, _last_futu_EMA_26)
			_last_futu_EMA_12 = _futu_EMA_12
			_last_futu_EMA_26 = _futu_EMA_26
			LIST_DIF.append(_futu_EMA_12 - _futu_EMA_26)
			if i > 0:
				LIST_D_DIF.append(LIST_DIF[i] - LIST_DIF[i-1])

			# RSI
			if i > 0:
				_rsi_temp1 = max(LIST_CLOSE[i] - LIST_CLOSE[i-1], 0)
				_rsi_temp2 = 1 if abs(LIST_CLOSE[i] - LIST_CLOSE[i-1])==0 else abs(LIST_CLOSE[i] - LIST_CLOSE[i-1])
			else:
				_rsi_temp1 = 0
				_rsi_temp2 = 1
				_last_rsi_temp1_sma = 0
				_last_rsi_temp2_sma = 1
			_rsi_temp1_sma = futu_sma(_rsi_temp1, 1, 12, _last_rsi_temp1_sma)
			_rsi_temp2_sma = futu_sma(_rsi_temp2, 1, 12, _last_rsi_temp2_sma)
			_last_rsi_temp1_sma = _rsi_temp1_sma
			_last_rsi_temp2_sma = _rsi_temp2_sma
			LIST_RSI12.append(_rsi_temp1_sma / _rsi_temp2_sma * 100)
			if i > 0:
				LIST_D_RSI12.append(LIST_RSI12[i] - LIST_RSI12[i-1])
			LIST_MA4_D_RSI12.append(futu_sma(LIST_D_RSI12[i], 1, 4, _last_ma4_d_rsi12))
			_last_ma4_d_rsi12 = LIST_MA4_D_RSI12[i]

			#Buy rating
			#ROUND2(SMA((100 - RSI)/1,3,2)/CLOSE*100,2))
			#SMA(X,N,M), such as Y=(X*M + Y'*(N-M))/N
			if i == 0:
				# _temp = 1 if LIST_CLOSE[0] == 0 else LIST_CLOSE[0]
				# _last_buy_rating_RSI = (100-LIST_RSI12[0])/_temp*100
				_last_buy_rating_RSI = LIST_RSI12[0]
				MAX_RSI_BR = LIST_RSI12[0]
				MIN_RSI_BR = LIST_RSI12[0]

			_SMA_buy_rating_RSI = futu_sma(LIST_RSI12[i], 2, 3, _last_buy_rating_RSI)
			_last_buy_rating_RSI = _SMA_buy_rating_RSI
			LIST_BUY_RATING.append(_SMA_buy_rating_RSI)
			# update min max
			if _SMA_buy_rating_RSI > MAX_RSI_BR:
				MAX_RSI_BR = _SMA_buy_rating_RSI
			if _SMA_buy_rating_RSI < MIN_RSI_BR:
				MIN_RSI_BR = _SMA_buy_rating_RSI

			# _SMA_buy_rating_RSI = futu_sma((100-LIST_RSI12[i]), 2, 3, _last_buy_rating_RSI)
			# _last_buy_rating_RSI = _SMA_buy_rating_RSI
			# _temp = 1 if LIST_CLOSE[i] == 0 else LIST_CLOSE[i]
			# LIST_BUY_RATING.append(_SMA_buy_rating_RSI/_temp*100)

		# 2nd for loop
		_last_dea = LIST_DIF[0]
		for i in range(0, MAX_INDEX):
			# MACD
			LIST_DEA.append(futu_ema(LIST_DIF[i], 9, _last_dea))
			_last_dea = LIST_DEA[i]
			if i > 0:
				LIST_D_DEA.append(LIST_DEA[i] - LIST_DEA[i-1])
			LIST_MACD.append(LIST_DIF[i] - LIST_DEA[i])

		# CCI
		_temp_sum_list = []
		for i in range(0, MAX_INDEX):
			_temp_sum = 0
			for _i in range(i, i-12, -1):
				_check_i = 0 if _i < 0 else _i
				_temp_sum += LIST_HLC[_check_i]
			_temp_sum_list.append(_temp_sum)
			if i == 0:
				MAX_cci_mean = _temp_sum
				MIN_cci_mean = _temp_sum
			# update min max
			if _temp_sum > MAX_cci_mean:
				MAX_cci_mean = _temp_sum
			if _temp_sum < MIN_cci_mean:
				MIN_cci_mean = _temp_sum
		df_stock['cci_mean'] = _temp_sum_list
		df_stock['cci_mean'] = df_stock['cci_mean'] / 12
		LIST_CCI_MEAN = df_stock['cci_mean'].copy().to_list()

		_temp_sum_list = []
		for i in range(0, MAX_INDEX):
			_temp_sum = 0
			for _i in range(i, i-12, -1):
				_check_i = 0 if _i < 0 else _i
				_temp_sum = _temp_sum + (LIST_HLC[_check_i] - LIST_CCI_MEAN[_check_i])
			_temp_sum_list.append(_temp_sum)
		df_stock['cci_avedev'] = _temp_sum_list
		df_stock['cci_avedev'] = df_stock['cci_avedev'] / 12
		df_stock['cci12'] = round((df_stock['HLC'] - df_stock['cci_mean']) / 0.015 / df_stock['cci_avedev'], 3)

		# PDI & MDI
		_temp_sum_list = []
		for i in range(0, MAX_INDEX):
			_temp_sum = 0
			for _i in range(i, i-14, -1):
				_check_i = 0 if _i < 0 else _i
				if LIST_HD[_check_i] > 0 and LIST_HD[_check_i] > LIST_LD[_check_i]:
					_temp_sum += LIST_HD[_check_i]
			_temp_sum_list.append(_temp_sum)
		df_stock['DMP'] = _temp_sum_list
		LIST_DMP = df_stock['DMP'].copy().to_list()
		_temp_sum_list = []
		for i in range(0, MAX_INDEX):
			_temp_sum = 0
			for _i in range(i, i-14, -1):
				_check_i = 0 if _i < 0 else _i
				if LIST_LD[_check_i] > 0 and LIST_LD[_check_i] > LIST_HD[_check_i]:
					_temp_sum += LIST_LD[_check_i]
			_temp_sum_list.append(_temp_sum)
		df_stock['DMM'] = _temp_sum_list
		LIST_DMM = df_stock['DMM'].copy().to_list()

		_temp_sum_list = []
		for i in range(0, MAX_INDEX):
			_temp_sum = 0
			for _i in range(i, i-14, -1):
				_check_i = 0 if _i < 0 else _i
				_temp_sum += max(max(LIST_HIGH[_check_i]-LIST_LOW[_check_i], abs(LIST_HIGH[_check_i]-LIST_CLOSE[max(0,_check_i-1)])), abs(LIST_LOW[_check_i]-LIST_CLOSE[max(0,_check_i-1)]))
			_temp_sum_list.append(_temp_sum)
		df_stock['TR1'] = _temp_sum_list
		LIST_TR1 = df_stock['TR1'].copy().to_list()

		_list_PDI_ema8 = []
		_list_MDI_ema8 = []
		for i in range(0, MAX_INDEX):
			if LIST_TR1[i] != 0:
				_list_PDI_ema8.append(futu_ema(LIST_DMP[i] * 100 / LIST_TR1[i], 8, (LIST_DMP[i] * 100 / LIST_TR1[i] if i==0 else _list_PDI_ema8[i-1])))
				_list_MDI_ema8.append(futu_ema(LIST_DMM[i] * 100 / LIST_TR1[i], 8, (LIST_DMM[i] * 100 / LIST_TR1[i] if i==0 else _list_MDI_ema8[i-1])))
			else:
				_list_PDI_ema8.append(0)
				_list_MDI_ema8.append(0)
		df_stock['PDI_ema8'] = _list_PDI_ema8
		df_stock['MDI_ema8'] = _list_MDI_ema8

		_boolean_buy_deltaMACD = False
		_boolean_buy_delta_DDIF = False
		_boolean_buy_delta_DDEA = False
		_boolean_buy_deltaRSI = False
		_boolean_buy_deltaCCI = False
		_boolean_buy_PDI_MDI = False
		_boolean_sell_deltaMACD = False
		_boolean_sell_delta_DDIF = False
		_boolean_sell_delta_DDEA = False
		_boolean_sell_deltaRSI = False
		_boolean_sell_deltaCCI = False
		_boolean_sell_deltaDIF = False
		_boolean_sell_ma4_d_rsi12 = False

		_debug_list_buy_deltaMACD = []
		_debug_list_buy_delta_DDIF = []
		_debug_list_buy_delta_DDEA = []
		_debug_list_buy_deltaRSI = []
		_debug_list_buy_deltaCCI = []
		_debug_list_buy_PDI_MDI = []

		_debuy_list_sell_deltaMACD = []
		_debuy_list_sell_delta_DDIF = []
		_debuy_list_sell_delta_DDEA = []
		_debuy_list_sell_deltaRSI = []
		_debuy_list_sell_deltaCCI = []
		_debuy_list_sell_deltaDIF = []
		_debuy_list_sell_ma4_d_rsi12 = []

		_list_state = []
		_list_state.append('Cash')

		_list_debug_buy_check = []
		_list_debug_sell_check = []

		LIST_CCI12 = df_stock['cci12'].copy().to_list()
		LIST_PDI_EMA8 = df_stock['PDI_ema8'].copy().to_list()
		LIST_MDI_EMA8 = df_stock['MDI_ema8'].copy().to_list()

		for i in range(0, MAX_INDEX):
			_boolean_buy_deltaMACD = LIST_MACD[i] > (LIST_MACD[i] if i==0 else LIST_MACD[i-1])
			_boolean_buy_delta_DDIF = LIST_D_DIF[i] > (LIST_D_DIF[i] if i==0 else LIST_D_DIF[i-1])
			_boolean_buy_delta_DDEA = LIST_D_DEA[i] > (LIST_D_DEA[i] if i==0 else LIST_D_DEA[i-1])
			_boolean_buy_deltaRSI = LIST_RSI12[i] > (LIST_RSI12[i] if i==0 else LIST_RSI12[i-1])
			_boolean_buy_deltaCCI = LIST_CCI12[i] > (LIST_CCI12[i] if i==0 else LIST_CCI12[i-1])
			_boolean_buy_PDI_MDI = LIST_PDI_EMA8[i] > LIST_MDI_EMA8[i]
			_debug_list_buy_deltaMACD.append(_boolean_buy_deltaMACD)
			_debug_list_buy_delta_DDIF.append(_boolean_buy_delta_DDIF)
			_debug_list_buy_delta_DDEA.append(_boolean_buy_delta_DDEA)
			_debug_list_buy_deltaRSI.append(_boolean_buy_deltaRSI)
			_debug_list_buy_deltaCCI.append(_boolean_buy_deltaCCI)
			_debug_list_buy_PDI_MDI.append(_boolean_buy_PDI_MDI)

			_boolean_sell_deltaMACD = LIST_MACD[i] < (LIST_MACD[i] if i==0 else LIST_MACD[i-1])
			_boolean_sell_delta_DDIF = LIST_D_DIF[i] < (LIST_D_DIF[i] if i==0 else LIST_D_DIF[i-1])
			_boolean_sell_delta_DDEA = LIST_D_DEA[i] < (LIST_D_DEA[i] if i==0 else LIST_D_DEA[i-1])
			_boolean_sell_deltaRSI = LIST_RSI12[i] < (LIST_RSI12[i] if i==0 else LIST_RSI12[i-1])
			_boolean_sell_deltaCCI = LIST_CCI12[i] < (LIST_CCI12[i] if i==0 else LIST_CCI12[i-1])
			_boolean_sell_deltaDIF = LIST_DIF[i] < (LIST_DIF[i] if i==0 else LIST_DIF[i-1])
			_boolean_sell_ma4_d_rsi12 = LIST_MA4_D_RSI12[i] < 0
			_debuy_list_sell_deltaMACD.append(_boolean_sell_deltaMACD)
			_debuy_list_sell_delta_DDIF.append(_boolean_sell_delta_DDIF)
			_debuy_list_sell_delta_DDEA.append(_boolean_sell_delta_DDEA)
			_debuy_list_sell_deltaRSI.append(_boolean_sell_deltaRSI)
			_debuy_list_sell_deltaCCI.append(_boolean_sell_deltaCCI)
			_debuy_list_sell_deltaDIF.append(_boolean_sell_deltaDIF)
			_debuy_list_sell_ma4_d_rsi12.append(_boolean_sell_ma4_d_rsi12)

			buy_check = _boolean_buy_deltaMACD and _boolean_buy_delta_DDIF and _boolean_buy_delta_DDEA and _boolean_buy_deltaRSI and _boolean_buy_deltaCCI and _boolean_buy_PDI_MDI
			sell_check = _boolean_sell_deltaMACD and _boolean_sell_delta_DDIF and _boolean_sell_delta_DDEA and _boolean_sell_deltaRSI and _boolean_sell_deltaCCI and _boolean_sell_deltaDIF and _boolean_sell_ma4_d_rsi12

			_list_debug_buy_check.append(buy_check)
			_list_debug_sell_check.append(sell_check)

			if i!=0:
				if (_last_state=='Cash' or _last_state=='Sell') and buy_check:
					_list_state.append('Buy')
				elif (_last_state=='Keep' or _last_state=='Buy') and sell_check:
					_list_state.append('Sell')
				elif _last_state=='Buy' or _last_state=='Keep':
					_list_state.append('Keep')
				else:
					_list_state.append('Cash')
			elif buy_check:
				_list_state[0] = 'Buy'


			_last_state = _list_state[-1]

		df_stock.sort_values(by='Date',inplace=True)
		#df_stock.to_csv("./stock_output.nosync/"+symbol+".csv", index=False)

		end_time = time.time()
		print(symbol, 'DONE', end_time - start_time)

		if debug:
			'''
			df_stock.loc[:, 'RSI12'] = LIST_RSI12
			df_stock.loc[:, 'RSI_MA3'] = LIST_BUY_RATING
			df_stock.loc[:, 'RSI_MA3_N'] = (df_stock['RSI_MA3'] - MIN_RSI_BR) / (MAX_RSI_BR - MIN_RSI_BR)
			df_stock.loc[:, 'RSI_MA3_N_CLOSE'] = df_stock['RSI_MA3_N'] / df_stock['Close']
			df_stock.loc[:, 'cci_mean_N'] = (df_stock['cci_mean'] - MIN_cci_mean) / (MAX_cci_mean - MIN_cci_mean)
			df_stock.loc[:, 'cci_mean_N_CLOSE'] = df_stock['cci_mean_N'] / df_stock['Close']
			df_stock.loc[:, 'cci_RSI_rating'] = ( df_stock['RSI_MA3_N'] * 2 + df_stock['cci_mean_N'] ) / 3
			df_stock.loc[:, 'cci_RSI_rating_CLOSE'] = df_stock['cci_RSI_rating'] / df_stock['Close']
			'''
			df_stock.loc[:, 'buy_condition_MACD'] = _debug_list_buy_deltaMACD
			df_stock.loc[:, 'buy_condition_DIF'] = _debug_list_buy_delta_DDIF
			df_stock.loc[:, 'buy_condition_DEA'] = _debug_list_buy_delta_DDEA
			df_stock.loc[:, 'buy_condition_RSI'] = _debug_list_buy_deltaRSI
			df_stock.loc[:, 'buy_condition_CCI'] = _debug_list_buy_deltaCCI
			df_stock.loc[:, 'buy_condition_PDI_and_MDI'] = _debug_list_buy_PDI_MDI
			df_stock.loc[:, 'sell_condition_MACD'] = _debuy_list_sell_deltaMACD
			df_stock.loc[:, 'sell_condition_DIF'] = _debuy_list_sell_delta_DDIF
			df_stock.loc[:, 'sell_condition_DEA'] = _debuy_list_sell_delta_DDEA
			df_stock.loc[:, 'sell_condition_RSI'] = _debuy_list_sell_deltaRSI
			df_stock.loc[:, 'sell_condition_CCI'] = _debuy_list_sell_deltaCCI
			df_stock.loc[:, 'sell_condition_DIF'] = _debuy_list_sell_deltaDIF
			df_stock.loc[:, 'sell_condition_ma_rsi'] = _debuy_list_sell_ma4_d_rsi12

			df_stock.loc[:, 'state'] = _list_state
			# return the full df_stock

			df_stock = df_stock.drop(columns=['cci_avedev', 'cci12', 'DMP', 'DMM', 'TR1', 'PDI_ema8', 'MDI_ema8','RSI_MA3','RSI_MA3_N','cci_mean_N'])
			output = df_stock.copy()
		else:
			# return a boolean for buy or sell
			buy_found = -1 * (buy_mark_day+1)
			for i in range(-1*buy_mark_day, 0):
				if _list_state[i] == 'Buy':
					buy_found = i
			sell_found = -1 * (buy_mark_day+1)
			for i in range(-1*buy_mark_day, 0):
				if _list_state[i] == 'Sell':
					sell_found = i

			output = buy_found > sell_found

		return output

def buyRating(
        symbol: str,
        buy_mark_day: int,
        price_above: float,
        volumn_above: float,
        INTERVAL: str,
        debug: bool,
) -> Union[pd.DataFrame, bool]:
        """Evaluate the buy rating indicator for a given symbol.

        Args:
        - symbol: Stock ticker to analyse.
        - buy_mark_day: Number of recent days to look back for a buy signal.
        - price_above: Minimum allowed closing price for the latest day.
        - volumn_above: Minimum allowed trading volume for the latest day.
        - INTERVAL: Data interval passed to ``load_stock_history``.
        - debug: When ``True`` return the full DataFrame of calculated values.

        Returns:
        - DataFrame with all computed columns when ``debug`` is ``True``.
        - Otherwise, boolean indicating whether a recent buy signal met the
          rating criteria within the last ``buy_mark_day`` days.

        Side Effects:
        - Downloads data via :func:`load_stock_history`.
        - Prints progress and execution time to stdout.
        """
        df_stock = load_stock_history(symbol, INTERVAL, decimals=7)

	if df_stock.empty:
		return False
	elif df_stock['Close'].iloc[-1] < price_above:		# check price higher than requirement
		return False
	elif df_stock['Volume'].iloc[-1] < volumn_above:	# check price higher than requirement
		return False
	elif len(df_stock.index) < 60:						# new stock within 60 trading day, SKIP
		return False
	else:
		start_time = time.time()
		MAX_INDEX = len(df_stock.index)

		df_stock.loc[:, 'HLC'] = round((df_stock['High'] + df_stock['Low'] + df_stock['Close']) / 3, 7)
		LIST_HIGH = df_stock['High'].copy().to_list()
		LIST_LOW = df_stock['Low'].copy().to_list()
		LIST_CLOSE = df_stock['Close'].copy().to_list()
		LIST_HLC = df_stock['HLC'].copy().to_list()

		LIST_HD = []
		LIST_LD = []
		for i in range(0, MAX_INDEX):
			LIST_HD.append(LIST_HIGH[i] - (LIST_HIGH[i] if i==0 else LIST_HIGH[i-1]))
			LIST_LD.append((LIST_LOW[i] if i==0 else LIST_LOW[i-1]) - LIST_LOW[i])

		df_stock.loc[:, 'cci_mean'] = 0
		df_stock.loc[:, 'cci_avedev'] = 0
		df_stock.loc[:, 'cci12'] = 0

		df_stock.loc[:, 'DMP'] = 0
		df_stock.loc[:, 'DMM'] = 0
		df_stock.loc[:, 'TR1'] = 0
		df_stock.loc[:, 'PDI_ema8'] = 0
		df_stock.loc[:, 'MDI_ema8'] = 0

		# init before for
		LIST_DIF = []							# MACD
		LIST_DEA = []							# MACD
		LIST_MACD = []							# MACD
		LIST_D_DIF = [0]						# MACD
		LIST_D_DEA = [0]						# MACD
		LIST_RSI12 = []							# RSI
		LIST_D_RSI12 = [0]						# RSI
		LIST_MA4_D_RSI12 = []					# RSI
		LIST_BUY_RATING = []					# KUSHI BUY_RATING

		# 1st for loop
		_last_futu_EMA_12 = LIST_CLOSE[0]		# MACD
		_last_futu_EMA_26 = LIST_CLOSE[0]		# MACD
		_last_ma4_d_rsi12 = LIST_D_RSI12[0]		# RSI
		for i in range(0, MAX_INDEX):
			# MACD
			_futu_EMA_12 = futu_ema(LIST_CLOSE[i], 12, _last_futu_EMA_12)
			_futu_EMA_26 = futu_ema(LIST_CLOSE[i], 26, _last_futu_EMA_26)
			_last_futu_EMA_12 = _futu_EMA_12
			_last_futu_EMA_26 = _futu_EMA_26
			LIST_DIF.append(_futu_EMA_12 - _futu_EMA_26)
			if i > 0:
				LIST_D_DIF.append(LIST_DIF[i] - LIST_DIF[i-1])

			# RSI
			if i > 0:
				_rsi_temp1 = max(LIST_CLOSE[i] - LIST_CLOSE[i-1], 0)
				_rsi_temp2 = 1 if abs(LIST_CLOSE[i] - LIST_CLOSE[i-1])==0 else abs(LIST_CLOSE[i] - LIST_CLOSE[i-1])
			else:
				_rsi_temp1 = 0
				_rsi_temp2 = 1
				_last_rsi_temp1_sma = 0
				_last_rsi_temp2_sma = 1
			_rsi_temp1_sma = futu_sma(_rsi_temp1, 1, 12, _last_rsi_temp1_sma)
			_rsi_temp2_sma = futu_sma(_rsi_temp2, 1, 12, _last_rsi_temp2_sma)
			_last_rsi_temp1_sma = _rsi_temp1_sma
			_last_rsi_temp2_sma = _rsi_temp2_sma
			LIST_RSI12.append(_rsi_temp1_sma / _rsi_temp2_sma * 100)
			if i > 0:
				LIST_D_RSI12.append(LIST_RSI12[i] - LIST_RSI12[i-1])
			LIST_MA4_D_RSI12.append(futu_sma(LIST_D_RSI12[i], 1, 4, _last_ma4_d_rsi12))
			_last_ma4_d_rsi12 = LIST_MA4_D_RSI12[i]

			#Buy rating
			#ROUND2(SMA((100 - RSI)/1,3,2)/CLOSE*100,2))
			#SMA(X,N,M), such as Y=(X*M + Y'*(N-M))/N
			if i == 0:
				# _temp = 1 if LIST_CLOSE[0] == 0 else LIST_CLOSE[0]
				# _last_buy_rating_RSI = (100-LIST_RSI12[0])/_temp*100
				_last_buy_rating_RSI = LIST_RSI12[0]

			_SMA_buy_rating_RSI = futu_sma(LIST_RSI12[i], 2, 3, _last_buy_rating_RSI)
			_last_buy_rating_RSI = _SMA_buy_rating_RSI
			LIST_BUY_RATING.append(_SMA_buy_rating_RSI)
			# _SMA_buy_rating_RSI = futu_sma((100-LIST_RSI12[i]), 2, 3, _last_buy_rating_RSI)
			# _last_buy_rating_RSI = _SMA_buy_rating_RSI
			# _temp = 1 if LIST_CLOSE[i] == 0 else LIST_CLOSE[i]
			# LIST_BUY_RATING.append(_SMA_buy_rating_RSI/_temp*100)

		# 2nd for loop
		_last_dea = LIST_DIF[0]
		for i in range(0, MAX_INDEX):
			# MACD
			LIST_DEA.append(futu_ema(LIST_DIF[i], 9, _last_dea))
			_last_dea = LIST_DEA[i]
			if i > 0:
				LIST_D_DEA.append(LIST_DEA[i] - LIST_DEA[i-1])
			LIST_MACD.append(LIST_DIF[i] - LIST_DEA[i])

		df_stock.loc[:, "LIST_D_DEA"] = LIST_D_DEA

		# CCI
		_temp_sum_list = []
		for i in range(0, MAX_INDEX):
			_temp_sum = 0
			for _i in range(i, i-12, -1):
				_check_i = 0 if _i < 0 else _i
				_temp_sum += LIST_HLC[_check_i]
			_temp_sum_list.append(_temp_sum)
		df_stock['cci_mean'] = _temp_sum_list
		df_stock['cci_mean'] = df_stock['cci_mean'] / 12
		LIST_CCI_MEAN = df_stock['cci_mean'].copy().to_list()

		_temp_sum_list = []
		for i in range(0, MAX_INDEX):
			_temp_sum = 0
			for _i in range(i, i-12, -1):
				_check_i = 0 if _i < 0 else _i
				_temp_sum = _temp_sum + (LIST_HLC[_check_i] - LIST_CCI_MEAN[_check_i])
			_temp_sum_list.append(_temp_sum)
		df_stock['cci_avedev'] = _temp_sum_list
		df_stock['cci_avedev'] = df_stock['cci_avedev'] / 12
		df_stock['cci12'] = round((df_stock['HLC'] - df_stock['cci_mean']) / 0.015 / df_stock['cci_avedev'], 3)

		# PDI & MDI
		_temp_sum_list = []
		for i in range(0, MAX_INDEX):
			_temp_sum = 0
			for _i in range(i, i-14, -1):
				_check_i = 0 if _i < 0 else _i
				if LIST_HD[_check_i] > 0 and LIST_HD[_check_i] > LIST_LD[_check_i]:
					_temp_sum += LIST_HD[_check_i]
			_temp_sum_list.append(_temp_sum)
		df_stock['DMP'] = _temp_sum_list
		LIST_DMP = df_stock['DMP'].copy().to_list()
		_temp_sum_list = []
		for i in range(0, MAX_INDEX):
			_temp_sum = 0
			for _i in range(i, i-14, -1):
				_check_i = 0 if _i < 0 else _i
				if LIST_LD[_check_i] > 0 and LIST_LD[_check_i] > LIST_HD[_check_i]:
					_temp_sum += LIST_LD[_check_i]
			_temp_sum_list.append(_temp_sum)
		df_stock['DMM'] = _temp_sum_list
		LIST_DMM = df_stock['DMM'].copy().to_list()

		_temp_sum_list = []
		for i in range(0, MAX_INDEX):
			_temp_sum = 0
			for _i in range(i, i-14, -1):
				_check_i = 0 if _i < 0 else _i
				_temp_sum += max(max(LIST_HIGH[_check_i]-LIST_LOW[_check_i], abs(LIST_HIGH[_check_i]-LIST_CLOSE[max(0,_check_i-1)])), abs(LIST_LOW[_check_i]-LIST_CLOSE[max(0,_check_i-1)]))
			_temp_sum_list.append(_temp_sum)
		df_stock['TR1'] = _temp_sum_list
		LIST_TR1 = df_stock['TR1'].copy().to_list()

		_list_PDI_ema8 = []
		_list_MDI_ema8 = []
		for i in range(0, MAX_INDEX):
			if LIST_TR1[i] != 0:
				_list_PDI_ema8.append(futu_ema(LIST_DMP[i] * 100 / LIST_TR1[i], 8, (LIST_DMP[i] * 100 / LIST_TR1[i] if i==0 else _list_PDI_ema8[i-1])))
				_list_MDI_ema8.append(futu_ema(LIST_DMM[i] * 100 / LIST_TR1[i], 8, (LIST_DMM[i] * 100 / LIST_TR1[i] if i==0 else _list_MDI_ema8[i-1])))
			else:
				_list_PDI_ema8.append(0)
				_list_MDI_ema8.append(0)
		df_stock['PDI_ema8'] = _list_PDI_ema8
		df_stock['MDI_ema8'] = _list_MDI_ema8

		_boolean_buy_deltaMACD = False
		_boolean_buy_delta_DDIF = False
		_boolean_buy_delta_DDEA = False
		_boolean_buy_deltaRSI = False
		_boolean_buy_deltaCCI = False
		_boolean_buy_PDI_MDI = False
		_boolean_sell_deltaMACD = False
		_boolean_sell_delta_DDIF = False
		_boolean_sell_delta_DDEA = False
		_boolean_sell_deltaRSI = False
		_boolean_sell_deltaCCI = False
		_boolean_sell_deltaDIF = False
		_boolean_sell_ma4_d_rsi12 = False

		_debug_list_buy_deltaMACD = []
		_debug_list_buy_delta_DDIF = []
		_debug_list_buy_delta_DDEA = []
		_debug_list_buy_deltaRSI = []
		_debug_list_buy_deltaCCI = []
		_debug_list_buy_PDI_MDI = []

		_debuy_list_sell_deltaMACD = []
		_debuy_list_sell_delta_DDIF = []
		_debuy_list_sell_delta_DDEA = []
		_debuy_list_sell_deltaRSI = []
		_debuy_list_sell_deltaCCI = []
		_debuy_list_sell_deltaDIF = []
		_debuy_list_sell_ma4_d_rsi12 = []

		_list_state = []
		_list_state.append('Cash')

		_list_debug_buy_check = []
		_list_debug_sell_check = []

		LIST_CCI12 = df_stock['cci12'].copy().to_list()
		LIST_PDI_EMA8 = df_stock['PDI_ema8'].copy().to_list()
		LIST_MDI_EMA8 = df_stock['MDI_ema8'].copy().to_list()

		for i in range(0, MAX_INDEX):
			_boolean_buy_deltaMACD = LIST_MACD[i] > (LIST_MACD[i] if i==0 else LIST_MACD[i-1])
			_boolean_buy_delta_DDIF = LIST_D_DIF[i] > (LIST_D_DIF[i] if i==0 else LIST_D_DIF[i-1])
			_boolean_buy_delta_DDEA = LIST_D_DEA[i] > (LIST_D_DEA[i] if i==0 else LIST_D_DEA[i-1])
			_boolean_buy_deltaRSI = LIST_RSI12[i] > (LIST_RSI12[i] if i==0 else LIST_RSI12[i-1])
			_boolean_buy_deltaCCI = LIST_CCI12[i] > (LIST_CCI12[i] if i==0 else LIST_CCI12[i-1])
			_boolean_buy_PDI_MDI = LIST_PDI_EMA8[i] > LIST_MDI_EMA8[i]
			_debug_list_buy_deltaMACD.append(_boolean_buy_deltaMACD)
			_debug_list_buy_delta_DDIF.append(_boolean_buy_delta_DDIF)
			_debug_list_buy_delta_DDEA.append(_boolean_buy_delta_DDEA)
			_debug_list_buy_deltaRSI.append(_boolean_buy_deltaRSI)
			_debug_list_buy_deltaCCI.append(_boolean_buy_deltaCCI)
			_debug_list_buy_PDI_MDI.append(_boolean_buy_PDI_MDI)

			_boolean_sell_deltaMACD = LIST_MACD[i] < (LIST_MACD[i] if i==0 else LIST_MACD[i-1])
			_boolean_sell_delta_DDIF = LIST_D_DIF[i] < (LIST_D_DIF[i] if i==0 else LIST_D_DIF[i-1])
			_boolean_sell_delta_DDEA = LIST_D_DEA[i] < (LIST_D_DEA[i] if i==0 else LIST_D_DEA[i-1])
			_boolean_sell_deltaRSI = LIST_RSI12[i] < (LIST_RSI12[i] if i==0 else LIST_RSI12[i-1])
			_boolean_sell_deltaCCI = LIST_CCI12[i] < (LIST_CCI12[i] if i==0 else LIST_CCI12[i-1])
			_boolean_sell_deltaDIF = LIST_DIF[i] < (LIST_DIF[i] if i==0 else LIST_DIF[i-1])
			_boolean_sell_ma4_d_rsi12 = LIST_MA4_D_RSI12[i] < 0
			_debuy_list_sell_deltaMACD.append(_boolean_sell_deltaMACD)
			_debuy_list_sell_delta_DDIF.append(_boolean_sell_delta_DDIF)
			_debuy_list_sell_delta_DDEA.append(_boolean_sell_delta_DDEA)
			_debuy_list_sell_deltaRSI.append(_boolean_sell_deltaRSI)
			_debuy_list_sell_deltaCCI.append(_boolean_sell_deltaCCI)
			_debuy_list_sell_deltaDIF.append(_boolean_sell_deltaDIF)
			_debuy_list_sell_ma4_d_rsi12.append(_boolean_sell_ma4_d_rsi12)

			buy_check = _boolean_buy_deltaMACD and _boolean_buy_delta_DDIF and _boolean_buy_delta_DDEA and _boolean_buy_deltaRSI and _boolean_buy_deltaCCI and _boolean_buy_PDI_MDI
			sell_check = _boolean_sell_deltaMACD and _boolean_sell_delta_DDIF and _boolean_sell_delta_DDEA and _boolean_sell_deltaRSI and _boolean_sell_deltaCCI and _boolean_sell_deltaDIF and _boolean_sell_ma4_d_rsi12

			_list_debug_buy_check.append(buy_check)
			_list_debug_sell_check.append(sell_check)

			if i!=0:
				if (_last_state=='Cash' or _last_state=='Sell') and buy_check:
					_list_state.append('Buy')
				elif (_last_state=='Keep' or _last_state=='Buy') and sell_check:
					_list_state.append('Sell')
				elif _last_state=='Buy' or _last_state=='Keep':
					_list_state.append('Keep')
				else:
					_list_state.append('Cash')
			elif buy_check:
				_list_state[0] = 'Buy'


			_last_state = _list_state[-1]

		df_stock.loc[:, 'buy_deltaMACD'] = _debug_list_buy_deltaMACD
		df_stock.loc[:, 'buy_delta_DDIF'] = _debug_list_buy_delta_DDIF
		df_stock.loc[:, 'buy_delta_DDEA'] = _debug_list_buy_delta_DDEA
		df_stock.loc[:, 'buy_deltaRSI'] = _debug_list_buy_deltaRSI
		df_stock.loc[:, 'buy_deltaCCI'] = _debug_list_buy_deltaCCI
		df_stock.loc[:, 'buy_PDI_MDI'] = _debug_list_buy_PDI_MDI
		df_stock.loc[:, 'sell_deltaMACD'] = _debuy_list_sell_deltaMACD
		df_stock.loc[:, 'sell_delta_DDIF'] = _debuy_list_sell_delta_DDIF
		df_stock.loc[:, 'sell_delta_DDEA'] = _debuy_list_sell_delta_DDEA
		df_stock.loc[:, 'sell_deltaRSI'] = _debuy_list_sell_deltaRSI
		df_stock.loc[:, 'sell_deltaCCI'] = _debuy_list_sell_deltaCCI
		df_stock.loc[:, 'sell_deltaDIF'] = _debuy_list_sell_deltaDIF
		df_stock.loc[:, 'sell_ma4_d_rsi12'] = _debuy_list_sell_ma4_d_rsi12
		df_stock.loc[:, 'state'] = _list_state

		df_stock.loc[:, 'buy_check'] = _list_debug_buy_check
		df_stock.loc[:, 'sell_check'] = _list_debug_sell_check
		df_stock.loc[:, 'buy_rating'] = LIST_BUY_RATING

		df_stock.sort_values(by='Date',inplace=True)
		#df_stock.to_csv("./stock_output.nosync/"+symbol+".csv", index=False)

		end_time = time.time()
		print(symbol, 'DONE', end_time - start_time)

		if debug:
			# return the full df_stock
			output = df_stock.copy()
		else:
			# return a boolean for buy or sell
			buy_found = -1 * (buy_mark_day+1)
			for i in range(-1*buy_mark_day, 0):
				if _list_state[i] == 'Buy':
					buy_found = i
			sell_found = -1 * (buy_mark_day+1)
			for i in range(-1*buy_mark_day, 0):
				if _list_state[i] == 'Sell':
					sell_found = i

			output = buy_found > sell_found

			# have buy mark, than check buy rating
			if output:
				output = False
				buy_found_rating = LIST_BUY_RATING[buy_found]
				second_buy_rating = 0.0
				count = 0
				second_buy_found = 0
				for i in range(len(LIST_BUY_RATING) - 1, len(LIST_BUY_RATING) - 51, -1):
					if _list_state[i] == 'Buy':
						count = count + 1
						if count == 2:
							second_buy_rating = LIST_BUY_RATING[i]
							second_buy_found = i
							break

				if second_buy_found != 0:
					output = (LIST_CLOSE[buy_found] > LIST_CLOSE[second_buy_found]) and (LIST_BUY_RATING[buy_found]*1.2 > LIST_BUY_RATING[second_buy_found])

		return output

def vol_up (symbol, buy_mark_day, price_above, volumn_above, INTERVAL, debug):
	# Get today's date
	end_date = datetime.date.today()
	# Subtract 100 years from today's date
	start_date = end_date - datetime.timedelta(days=365*100)
	# Now, download the stock data within this range
	df_stock = yf.download(symbol, start=start_date, end=end_date, interval=INTERVAL)

	if df_stock.empty:
		return False
	elif df_stock['Close'].iloc[-1] < price_above:		# check price higher than requirement
		return False
	elif df_stock['Volume'].iloc[-1] < volumn_above:		# check price higher than requirement
		return False
	elif len(df_stock.index) < 10:						# new stock within 10 trading day, SKIP
		return False
	else:
		start_time = time.time()
		# Step 2: Store data in a Pandas DataFrame
		df_stock.reset_index(inplace = True)

		MAX_INDEX = len(df_stock.index)

		#df_stock.insert(len(df_stock.columns), 'Open', 0.0)
		df_stock['Low'] = df_stock['Low'].round(3)
		df_stock['High'] = df_stock['High'].round(3)
		df_stock['Close'] = df_stock['Close'].round(3)
		df_stock['Open'] = df_stock['Open'].round(3)

		LIST_LOW = df_stock['Low'].copy().to_list()
		LIST_HIGH = df_stock['High'].copy().to_list()
		LIST_CLOSE = df_stock['Close'].copy().to_list()
		LIST_OPEN = df_stock['Open'].copy().to_list()
		LIST_VOL = df_stock['Volume'].copy().to_list()

		# 1) 今日 vol > vol MA 50 * 1.5
		LIST_MA_VOL50 = []
		_last_futu_SMA_VOL50 = LIST_VOL[0]
		VOLMA_PERCENTAGE = 1.5
		LIST_MA_VOL_CHECK = []
		# 2) vol buy power > sell power
		LIST_BUYPOWER_CHECK = []
		# 3) 今日 vol > 昨日 vol * 1.75
		VOLUP_PERCENTAGE = 1.75
		LIST_VOLUP_CHECK = []
		# 4) 52 week new high
		HHV_52 = []
		LIST_NEW_HGIH = []

		# first for loop
		for i in range(0, MAX_INDEX):
			# 今日 vol > vol MA 50 * 1.5
			#    vol ma 50
			_futu_SMA_VOL50 = futu_ema(LIST_VOL[i], 50, _last_futu_SMA_VOL50)
			_last_futu_SMA_VOL50 = _futu_SMA_VOL50
			LIST_MA_VOL50.append(_futu_SMA_VOL50)

			# 4) 52 week new high
			_high = LIST_HIGH[i]
			if i > 0 and i < 51:
				_temp_high = HHV_52[-1]
				if _temp_high > _high:
					_high = _temp_high
			elif i != 0:
				extra_one = i - 51 - 1
				if LIST_HIGH[extra_one] < HHV_52[-1]:
					_high = HHV_52[-1]
				else:
					for j in range(-51, 0):
						_temp_high = LIST_HIGH[0] if i+j<0 else LIST_HIGH[i+j]
						if _temp_high > _high:
							_high = _temp_high
			HHV_52.append(_high)

		# second for loop
		for i in range(0, MAX_INDEX):
			# 今日 vol > vol MA 50 * 1.5
			_temp_Check = LIST_VOL[i] > LIST_MA_VOL50[i] * VOLMA_PERCENTAGE
			LIST_MA_VOL_CHECK.append(_temp_Check)

			# 2) vol buy power > sell power
			#    BUY_POWER:= C - L;
			#    SELL_POWER:= H - C;
			#    TOTAL_POWER:= BUY_POWER + SELL_POWER;
			#    BUY_POWER_RATIO:= BUY_POWER / TOTAL_POWER;
			#    SELL_POWER_RATIO:= SELL_POWER / TOTAL_POWER;
			buy = LIST_CLOSE[i] - LIST_LOW[i]
			sell = LIST_HIGH[i] - LIST_CLOSE[i]
			total = buy + sell
			buyPower = 0 if total == 0 else buy / total
			sellPower = 0 if total == 0 else sell / total
			_temp_Check = buyPower > sellPower
			LIST_BUYPOWER_CHECK.append(_temp_Check)

			# 3) 今日 vol > 昨日 vol * 1.75
			_temp_Check = LIST_VOL[i] > (LIST_VOL[i] if i==0 else LIST_VOL[i-1]) * VOLUP_PERCENTAGE
			LIST_VOLUP_CHECK.append(_temp_Check)

			# 4) 52 week new high
			_temp_Check = LIST_HIGH[i] = HHV_52[i]
			LIST_NEW_HGIH.append(_temp_Check)

		LIST_STATE = []
		for i in range(0, MAX_INDEX):
			_temp_state = (LIST_MA_VOL_CHECK[i] and
				LIST_BUYPOWER_CHECK[i] and 
				LIST_VOLUP_CHECK[i] and 
				LIST_NEW_HGIH[i])

			LIST_STATE.append(_temp_state)

		end_time = time.time()
		print(symbol, 'DONE', end_time - start_time)

		if debug:
			# return the full df_stock
			df_stock.loc[:, 'STATE'] = LIST_STATE
			df_stock.loc[:, 'MA_VOL_CHECK'] = LIST_MA_VOL_CHECK
			df_stock.loc[:, 'BUYPOWER_CHECK'] = LIST_BUYPOWER_CHECK
			df_stock.loc[:, 'NEW_HGIH'] = LIST_NEW_HGIH
			output = df_stock.copy()
		else:
			# return a boolean for TRUE
			output = False
			for i in range(-1*buy_mark_day, 0):
				if LIST_STATE[i]:
					output = True

		return output

def mm_stage2 (symbol, buy_mark_day, price_above, volumn_above, INTERVAL, debug):
	# Get today's date
	end_date = datetime.date.today()
	# Subtract 100 years from today's date
	start_date = end_date - datetime.timedelta(days=365*100)
	# Now, download the stock data within this range
	df_stock = yf.download(symbol, start=start_date, end=end_date, interval=INTERVAL)

	if df_stock.empty:
		return False
	elif df_stock['Close'].iloc[-1] < price_above:		# check price higher than requirement
		return False
	elif df_stock['Volume'].iloc[-1] < volumn_above:	# check price higher than requirement
		return False
	elif len(df_stock.index) < 10:						# new stock within 10 trading day, SKIP
		return False
	else:
		start_time = time.time()
		# Step 2: Store data in a Pandas DataFrame
		df_stock.reset_index(inplace = True)

		MAX_INDEX = len(df_stock.index)

		#df_stock.insert(len(df_stock.columns), 'Open', 0.0)
		df_stock['Low'] = df_stock['Low'].round(3)
		df_stock['High'] = df_stock['High'].round(3)
		df_stock['Close'] = df_stock['Close'].round(3)
		df_stock['Open'] = df_stock['Open'].round(3)

		LIST_LOW = df_stock['Low'].copy().to_list()
		LIST_HIGH = df_stock['High'].copy().to_list()
		LIST_CLOSE = df_stock['Close'].copy().to_list()
		LIST_OPEN = df_stock['Open'].copy().to_list()
		LIST_VOL = df_stock['Volume'].copy().to_list()

		# 1) close > 150 day ma
		CHECK_1 = []
		LIST_MA_150 = []
		_last_futu_SMA_150 = LIST_CLOSE[0]
		# 2) close > 200 day ma
		CHECK_2 = []
		LIST_MA_200 = []
		_last_futu_SMA_200 = LIST_CLOSE[0]
		# 3) 150 day ma > 200 day ma
		CHECK_3 = []
		# 4）200 day ma trending up for at least a month
		_list_200ma = []
		CHECK_4 = []
		# 5）50 day ma > 150 day ma
		CHECK_5 = []
		LIST_MA_50 = []
		_last_futu_SMA_50 = LIST_CLOSE[0]
		# 6）50 day ma > 200 day ma
		CHECK_6 = []
		# 7）close > 50 day ma
		CHECK_7 = []
		# 8）close > 30% of its 52-week low
		CHECK_8 = []
		LLV_52 = []
		# 9）close <= 52-week high and close >= 75% of 52-week high
		CHECK_9 = []
		HHV_52 = []

		# first for loop
		for i in range(0, MAX_INDEX):
			# (5)(6)(7) 50 MA
			_futu_SMA_50 = futu_ema(LIST_CLOSE[i], 50, _last_futu_SMA_50)
			_last_futu_SMA_50 = _futu_SMA_50
			LIST_MA_50.append(_futu_SMA_50)

			# (1)(3)(5) 150 MA
			_futu_SMA_150 = futu_ema(LIST_CLOSE[i], 150, _last_futu_SMA_150)
			_last_futu_SMA_150 = _futu_SMA_150
			LIST_MA_150.append(_futu_SMA_150)

			# (2)(3)(4)(6) 200 MA
			_futu_SMA_200 = futu_ema(LIST_CLOSE[i], 200, _last_futu_SMA_200)
			_last_futu_SMA_200 = _futu_SMA_200
			LIST_MA_200.append(_futu_SMA_200)

			# (8) 52 week new low
			_low = LIST_LOW[i]
			if i > 0 and i < 51:
				_temp_low = LLV_52[-1]
				if _temp_low > _low:
					_low = _temp_low
			elif i != 0:
				extra_one = i - 51 - 1
				if LIST_LOW[extra_one] < LLV_52[-1]:
					_low = LLV_52[-1]
				else:
					for j in range(-51, 0):
						_temp_low = LIST_LOW[0] if i+j<0 else LIST_LOW[i+j]
						if _temp_low > _low:
							_low = _temp_low
			LLV_52.append(_low)

			# (9) 52 week new high
			_high = LIST_HIGH[i]
			if i > 0 and i < 51:
				_temp_high = HHV_52[-1]
				if _temp_high > _high:
					_high = _temp_high
			elif i != 0:
				extra_one = i - 51 - 1
				if LIST_HIGH[extra_one] < HHV_52[-1]:
					_high = HHV_52[-1]
				else:
					for j in range(-51, 0):
						_temp_high = LIST_HIGH[0] if i+j<0 else LIST_HIGH[i+j]
						if _temp_high > _high:
							_high = _temp_high
			HHV_52.append(_high)

		# second for loop
		for i in range(0, MAX_INDEX):
			# 1) close > 150 day ma
			_temp_1 = LIST_CLOSE[i]
			_temp_2 = LIST_MA_150[i]
			CHECK_1.append(_temp_1 > _temp_2)

			# 2) close > 200 day ma
			_temp_1 = LIST_CLOSE[i]
			_temp_2 = LIST_MA_200[i]
			CHECK_2.append(_temp_1 > _temp_2)

			# 3) 150 day ma > 200 day ma
			_temp_1 = LIST_MA_150[i]
			_temp_2 = LIST_MA_200[i]
			CHECK_3.append(_temp_1 > _temp_2)

			# 4）200 day ma trending up for at least a month
			if i <= 20:
				CHECK_4.append(False)
			else:
				_list_200ma = []
				for j in range(-20, 0):
					_temp = i -20 - 1
					_list_200ma.append(LIST_MA_200[_temp])
				result = linear_regression_trend(_list_200ma)

				_slope = result[0]
				_r_square = result[1] # 0 to 1, 1 means all points lay on the line
				CHECK_4.append(_slope > 0 and _r_square > 0.5)

			# 5）50 day ma > 150 day ma
			_temp_1 = LIST_MA_50[i]
			_temp_2 = LIST_MA_150[i]
			CHECK_5.append(_temp_1 > _temp_2)

			# 6）50 day ma > 200 day ma
			_temp_1 = LIST_MA_50[i]
			_temp_2 = LIST_MA_200[i]
			CHECK_6.append(_temp_1 > _temp_2)

			# 7）close > 50 day ma
			_temp_1 = LIST_CLOSE[i]
			_temp_2 = LIST_MA_50[i]
			CHECK_7.append(_temp_1 > _temp_2)

			# 8）close > 30% of its 52-week low
			_temp_1 = LIST_CLOSE[i]
			_temp_2 = LLV_52[i]
			CHECK_8.append(_temp_1 > 0.3*_temp_2)

			# 9）close <= 52-week high and close >= 75% of 52-week high
			_temp_1 = LIST_CLOSE[i]
			_temp_2 = HHV_52[i]
			CHECK_9.append(_temp_1 <= _temp_2 and _temp_1 > 0.75*_temp_2)

		LIST_STATE = []
		for i in range(0, MAX_INDEX):
			_temp_state = (CHECK_1[i] and
				CHECK_2[i] and 
				CHECK_3[i] and 
				CHECK_4[i] and 
				CHECK_5[i] and 
				CHECK_6[i] and 
				CHECK_7[i] and 
				CHECK_8[i] and 
				CHECK_9[i])

			LIST_STATE.append(_temp_state)

		end_time = time.time()
		print(symbol, 'DONE', end_time - start_time)

		if debug:
			# return the full df_stock
			df_stock.loc[:, 'STATE'] = LIST_STATE
			df_stock.loc[:, '>150'] = CHECK_1
			df_stock.loc[:, '>200'] = CHECK_2
			df_stock.loc[:, '150>200'] = CHECK_3
			df_stock.loc[:, '200up'] = CHECK_4
			df_stock.loc[:, '50>150'] = CHECK_5
			df_stock.loc[:, '50>200'] = CHECK_6
			df_stock.loc[:, '>50'] = CHECK_7
			df_stock.loc[:, '>LLV_52*0.3'] = CHECK_8
			df_stock.loc[:, '>HHV_52*0.75,<=HHV'] = CHECK_9
			output = df_stock.copy()
		else:
			# return a boolean for TRUE
			output = False
			for i in range(-1*buy_mark_day, 0):
				if LIST_STATE[i]:
					output = True

		return output

