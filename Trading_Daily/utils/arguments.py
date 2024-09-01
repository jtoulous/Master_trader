import pandas as pd
import yfinance as yf

def GetArg(type):
    if type == 'lifespan':
        return 5
    elif type == 'risk':
        return 0.3
    elif type == 'profit':
        return 0.9
    elif type == 'atr':
        return 14
    elif type == 'ema':
        return 50
    elif type == 'rsi':
        return 14
    elif type == 'sto':
        return [14, 3]
    elif type == 'sma':
        return 50
    elif type == 'wma':
        return 20
    elif type == 'dmi':
        return 14
    elif type == 'blg':
        return [20, 2]
    elif type == 'macd':
        return [12, 26, 9]
    elif type == 'cci':
        return 20
    elif type == 'ppo':
        return [12, 26, 9]


def GetOpen(args, dataframe, crypto):
    if args.date is not None:
        date = pd.to_datetime(args.date, format='%d/%m/%Y')
    else:
        date = dataframe.iloc[-1]['DATETIME'] + pd.DateOffset(days=1)
    data = yf.download(crypto, start=date, interval='1d')
    if len(data) < 1:    
        return dataframe.iloc[-1]['CLOSE']
    return data.iloc[0]['Open']

def GetDate(args, dataframe):
    if args.date is not None:
        return pd.to_datetime(args.date, format='%d/%m/%Y')
    return dataframe.iloc[-1]['DATETIME'] + pd.DateOffset(days=1)

