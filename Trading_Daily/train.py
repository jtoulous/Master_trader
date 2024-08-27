import os
import argparse as ap

from models.models import GradientBoosting, LogReg, MLP, RFClassifier, XGB
from utils.preprocessing import preprocessing_train
from utils.tools import printLog, printError, printHeader


def parsing():
    parser = ap.ArgumentParser(
        prog='trading algo',
        description='predictive model for trading'
    )
    parser.add_argument('-BTCUSD', type=str, default=None, help='BTCUSD datafile')
    parser.add_argument('-ETHUSD', type=str, default=None, help='ETHUSD datafile')
    parser.add_argument('-BNBUSD', type=str, default=None, help='BNBUSD datafile')

    parser.add_argument('-lifespan', type=int, default=5, help='lifespan of the trade in days')
    parser.add_argument('-risk', type=float, default=0.3, help='percentage of capital for the stop-loss')
    parser.add_argument('-profit', type=float, default=0.9, help='percentage of capital for the take-profit')
    parser.add_argument('-atr', type=int, default=14, help='periods used for calculating ATR')
    parser.add_argument('-ema', type=int, default=50, help='periods used for calculating EMA')
    parser.add_argument('-rsi', type=int, default=14, help='periods used for calculating RSI')
    parser.add_argument('-sto', type=int, default=[14, 3], nargs=2, help='periods used for calculating STO')
    parser.add_argument('-sma', type=int, default=50, help='periods used for calculating SMA')
    parser.add_argument('-wma', type=int, default=20, help='periods used for calculating WMA')
    parser.add_argument('-dmi', type=int, default=14, help='periods used for calculating DMI')
    parser.add_argument('-blg', type=int, default=[20, 2], nargs=2, help='periods and nb_stddev used for calculating Bollingers bands')
    parser.add_argument('-macd', type=int, default=[12, 26, 9], nargs=3, help='periods(short, long, signal) used for calculating MACD')
    parser.add_argument('-cci', type=int, default=20, help='periods used for calculating CCI')
    parser.add_argument('-ppo', type=int, default=[12, 26, 9], nargs=3, help='periods(short, long, signal) used for calculating PPO')
    return parser.parse_args()


def trainModels(dataframe, currency_pair):
    printLog('\nTRAINING GRADIENT BOOSTING MODEL...')
    GradientBoosting(dataframe, currency_pair)
    printLog('GRADIENT BOOSTING MODEL DONE\n')

    printLog('TRAINING MLP MODEL...')
    MLP(dataframe, currency_pair)
    printLog('MLP MODEL DONE\n')

    printLog('TRAINING LOGISTIC REGRESSION MODEL...')
    LogReg(dataframe, currency_pair)
    printLog('LOGISTIC REGRESSION MODEL DONE\n')

    printLog('TRAINING RANDOM FOREST CLASSIFIER MODEL...')
    RFClassifier(dataframe, currency_pair)
    printLog('RANDOM FOREST CLASSIFIER MODEL DONE\n')

    printLog('TRAINING XGB MODEL...')
    XGB(dataframe, currency_pair)
    printLog('XGB MODEL DONE\n')


if __name__ == '__main__':
    try:
        args = parsing()

        if args.BTCUSD is not None:
            printHeader('BTCUSD')
            dataframe = preprocessing_train('BTCUSD', args, args.BTCUSD)
            trainModels(dataframe, 'BTCUSD')

        if args.ETHUSD is not None:
            printHeader('ETHUSD')
            dataframe = preprocessing_train('ETHUSD', args, args.ETHUSD)
            trainModels(dataframe, 'ETHUSD')
        
        if args.BNBUSD is not None:
            printHeader('BNBUSD')
            dataframe = preprocessing_train('BNBUSD', args, args.BNBUSD)
            trainModels(dataframe, 'BNBUSD')



    except Exception as error:
        printError(error)