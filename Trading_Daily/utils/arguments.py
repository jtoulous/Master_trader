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
        'LTC-USD',
        'BCH-USD',
        'NEAR-USD',
        'UNI7083-USD',
    ]
    return active_cryptos

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

def GetArg(arg_type):
    if arg_type == 'lifespan':
        return 10
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


def GetRP(crypto, arg_type):
    if crypto == 'BTC-USD':
        return 0.6 if arg_type == 'R' else 1.5
    if crypto == 'ETH-USD':
        return 0.6 if arg_type == 'R' else 1.5
    if crypto == 'SOL-USD':
        return 0.6 if arg_type == 'R' else 0.9
    if crypto == 'BNB-USD':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'ADA-USD':
        return 0.6 if arg_type == 'R' else 0.9
    if crypto == 'LINK-EUR':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'AVAX-USD':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'DOGE-USD':
        return 0.6 if arg_type == 'R' else 0.9
    if crypto == 'DOT-USD':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'TRX-EUR':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'XRP-USD':
        return 0.6 if arg_type == 'R' else 0.9
    if crypto == 'LTC-USD':
        return 0.6 if arg_type == 'R' else 1.5
    if crypto == 'BCH-USD':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'NEAR-USD':
        return 0.6 if arg_type == 'R' else 1.2
    if crypto == 'UNI7083-USD':
        return 0.6 if arg_type == 'R' else 1.2


def GetFeatures():
    features = [
        'OPEN',
        'DAY',
        'ATR',
        'EMA',
        'RSI',
        'K(sto)',
        'D(sto)',
        'SMA',
        'WMA',
        'DM+',
        'DM-',
        'ADX',
        'U-BAND',
        'L-BAND',
        'MACD_LINE',
        'MACD_SIGNAL',
        'MACD_HISTO',
        'Hilbert_Transform',
        'CCI',
        'PPO_LINE',
        'PPO_SIGNAL',
        'ROC',
        'ATR_Lagged',
        'EMA_Lagged',
        'RSI_Lagged',
        'Momentum',
        'MACD_Difference',
        'Bollinger_Width',
        'EMA_SMA_Ratio',
#        'VOLUME'
    ]
    return features


def UpdateArgs(args, crypto):
    args.risk = GetRP(crypto, 'R')
    args.profit = GetRP(crypto, 'P')
    return args