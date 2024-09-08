import pandas as pd
import yfinance as yf
import subprocess

from utils.log import printLog, printError
from utils.arguments import ActiveCryptos
from utils.dataframe import ReadDf, CleanDf

if __name__ == '__main__':
    try:
        cryptos = ActiveCryptos()
        for crypto in cryptos:
            df = ReadDf(f'{crypto}/{crypto}.csv')
            data_df = yf.download(crypto, start=df['DATETIME'].iloc[-5].date(), interval='1d')
            data_df = CleanDf(data_df)

            df.set_index('DATETIME', inplace=True)
            data_df.set_index('DATETIME', inplace=True)
            df = data_df.combine_first(df).sort_index()
            df.to_csv(f'{crypto}/{crypto}.csv')


    except Exception as error:
        printError(error)