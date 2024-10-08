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
            df = ReadDf(f'CRYPTOS/{crypto}/{crypto}.csv')
            data_df = yf.download(crypto, start=df['DATETIME'].iloc[-5].date(), interval='1d')
            data_df = CleanDf(data_df)

            df.set_index('DATETIME', inplace=True)
            data_df.set_index('DATETIME', inplace=True)
            df = data_df.combine_first(df).sort_index()
            df.to_csv(f'CRYPTOS/{crypto}/{crypto}.csv')

        printLog('\n===> Updating raw files...')
        subprocess.run(['python', 'split_in_years.py'])

        printLog('===> Updating test files...\n')
        subprocess.run(['python', 'build_test_files.py'])

    except Exception as error:
        printError(error)