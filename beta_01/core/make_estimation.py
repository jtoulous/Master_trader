import pandas as pd
import argparse as ap
from datetime import datetime

from utils.log import printLog, printError, printHeader
from utils.dataframe import ReadDf
from utils.estimate import Estimate
from utils.arguments import ActiveCryptos, GetCryptoFile

def parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('-crypto', type=str, default=None, help='crypto to make prediction on')
    parser.add_argument('-date', type=str, default=datetime.today().strftime('%d/%m/%Y'), help='date for estimation')
    parser.add_argument('-t', type=str, default='all', help='type of estimation(HIGH, LOW, CLOSE)')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parsing()
        crypto_list = [args.crypto] if args.crypto is not None else ActiveCryptos()
        estimation_type = args.t
        printLog(f'\n========================   {args.date}   ========================\n')

        for crypto in crypto_list:
            df = ReadDf(GetCryptoFile(crypto))
            printHeader(crypto)
            if estimation_type == 'all':
                printLog(f' LOW ==> {Estimate(df, args.date, "LOW")}')
                printLog(f' HIGH ==> {Estimate(df, args.date, "HIGH")}')
                printLog(f' CLOSE ==> {Estimate(df, args.date, "CLOSE")}')

            elif estimation_type == 'HIGH':
                printLog(f' HIGH ==> {Estimate(df, args.date, "HIGH")}')
            
            elif estimation_type == 'LOW':
                printLog(f' LOW ==> {Estimate(df, args.date, "LOW")}')

            elif estimation_type == 'CLOSE':
                printLog(f' CLOSE ==> {Estimate(df, args.date, "CLOSE")}')


    except Exception as error:
        printError(error)