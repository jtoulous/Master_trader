import pandas as pd
import yfinance as yf


def ActiveCryptos():
    active_cryptos = [
        'BTC-USD',
        'ETH-USD',
        'SOL-USD',
        'BNB-USD',
        'ADA-USD',
        'LINK-EUR',
        'AVAX-USD',
        'DOGE-USD',
        'DOT-USD',
        'TRX-EUR',
        'XRP-USD',
        'LTC-USD'
    ]
    return active_cryptos


def GetArg(arg_type):
    if arg_type == 'lifespan':
        return 20
    elif arg_type == 'risk':
        return 0.7
    elif arg_type == 'profit':
        return 1.8
    elif arg_type == 'atr':
        return 14
    elif arg_type == 'ema':
        return 50
    elif arg_type == 'rsi':
        return 14
    elif arg_type == 'sto':
        return [14, 3]
    elif arg_type == 'sma':
        return 50
    elif arg_type == 'wma':
        return 20
    elif arg_type == 'dmi':
        return 14
    elif arg_type == 'blg':
        return [20, 2]
    elif arg_type == 'macd':
        return [12, 26, 9]
    elif arg_type == 'cci':
        return 20
    elif arg_type == 'ppo':
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

def GetCryptoFile(crypto, file_type='default'):
    if file_type == 'default':
        return f'data/{crypto}/{crypto}.csv'
    if file_type == 'test train':
        return f'data/{crypto}/test_train.csv'
    if file_type == 'test predict':
        return f'data/{crypto}/test_predict.csv'