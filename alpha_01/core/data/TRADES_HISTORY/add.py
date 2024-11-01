import pandas as pd

from utils.log import printError, printLog
from utils.dataframe import ReadDf


if __name__ == '__main__':
    try:
        df = ReadDf('Trades_History.csv')
        date = input('Date(d/m/Y) ==> ')
        size = input('Size ==> ')
        price = input('Price ==> ')
        tp = input('TP ==> ')
        sl = input('SL ==> ')
        potential_win = input('Potential Win ==> ')
        potential_lose = input('Potential Lose ==> ')
        result = input('Result ==> ')

    except Exception as error:
        printError(error)