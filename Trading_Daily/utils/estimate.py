import pandas as pd

from .indicators import ATR, SMA, EMA, RSI, Bollinger_bands, MACD


def EstimateLow(dataframe, args):
    df = dataframe.copy()
    df['LABEL'] = df['LOW'].shift(-1)
    df = df.dropna(subset=['LABEL'])
    df = ATR(df, args.atr)
    df = SMA(dataframe, args.sma)
    df = EMA(dataframe, args.ema)
    df = RSI(dataframe, args.rsi)
    df = Bollinger_bands(dataframe, args.blg)
    df = MACD(dataframe, args.macd)



#def EstimateHigh(dataframe):
#
#def EstimateClose(dataframe):