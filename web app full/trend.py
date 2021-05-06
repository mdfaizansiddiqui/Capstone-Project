from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from datetime import timedelta

def SMA(df):
    days = 15
    df['SMA'] = df['Adj Close'].rolling(window = days).mean()
    return df

def MACD(df):
    exp1 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = exp3
    return df

def stochastic_K(df, k = 14, d = 3):
    low_min  = df['Low'].rolling( window = k ).min()
    high_max = df['High'].rolling( window = k ).max()
    # Fast Stochastic
    df['k'] = 100 * (df['Adj Close'] - low_min)/(high_max - low_min)
    # Slow stochastic
    df['d'] = df['k'].rolling(window = d).mean()
    return df


def computeRSI (data, time_window = 14):
    diff = data.diff(1).dropna() # diff in one field(one day)
    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

def get_technical_indicators(df):
    df = SMA(df)
    df = MACD(df)
    df = stochastic_K(df)
    df['RSI'] = computeRSI(df['Adj Close'])
    return df


def get_stock_data(start_date, end_date, stock):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if start_date.weekday() in [6, 7] or end_date.weekday() in [6, 7]:
        print("Date is Invalid!!")
        return -1
    start_new_date = start_date - timedelta(days = 30)
    start = str(start_new_date.year) +"-" + str(start_new_date.month) +"-" +str(start_new_date.day)
    end = str(end_date.year) +"-" + str(end_date.month) +"-" +str(end_date.day)
    df = data.DataReader(stock,'yahoo', start, end)
    df = get_technical_indicators(df)
#     df = normalize(df)
    return df.iloc[len(df)-1]


def get_trend(stock_data, stock_name):
    file_name = stock_name+"_trend"
    model = pickle.load(open("amazon_trend", "rb"))
    stock_data = np.array([stock_data['SMA'], stock_data['k'], stock_data['d'], stock_data['RSI'], stock_data['MACD']])
    stock_data = stock_data.reshape(1,5)
    pred = model.predict(stock_data)
    return pred[-1]