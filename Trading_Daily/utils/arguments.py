import pandas as pd
import yfinance as yf

#def GetArg(type):
#    if type == 'date':
#        return GetDate()
    

def GetOpen(args, dataframe, crypto):
    if args.date is not None:
        date = pd.to_datetime(args.date, format='%d/%m/%Y')
    else:
        date = dataframe.iloc[-1]['DATETIME'] + pd.DateOffset(days=1)
    data = yf.download(crypto, start=date, interval='1d')
    return data.iloc[0]['Open']

def GetDate(args, dataframe):
    if args.date is not None:
        return pd.to_datetime(args.date, format='%d/%m/%Y')
    return dataframe.iloc[-1]['DATETIME'] + pd.DateOffset(days=1)