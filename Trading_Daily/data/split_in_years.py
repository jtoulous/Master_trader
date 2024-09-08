import pandas as pd

from utils.log import printError
from utils.arguments import ActiveCryptos
from utils.dataframe import ReadDf

if __name__ == '__main__':
    try:
        cryptos = ActiveCryptos()
        for crypto in cryptos:
            df = ReadDf(f'{crypto}/{crypto}.csv')
            years = df['DATETIME'].dt.year.unique()

            for year in years:
                df_year = df[df['DATETIME'].dt.year == year].copy()
                df_year.to_csv(f'{crypto}/raw/{crypto}_{str(year)}.csv', index=False)

    except Exception as error:
        printError(error)