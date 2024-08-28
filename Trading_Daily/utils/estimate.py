import pandas as pd

from .indicators import ATR, SMA, EMA, RSI, MACD, DMI
from .indicators import Bollinger_bands, STO, ROC

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
    df = STO(dataframe, args.sto)
    df = ROC(dataframe)
    df = DMI(dataframe, args.dmi)
    df['GROWTH'] = (df['CLOSE'] - df['OPEN']) / df['OPEN'] * 100
    
    scaler = StandardScaler()


#def EstimateHigh(dataframe):
#
#def EstimateClose(dataframe):