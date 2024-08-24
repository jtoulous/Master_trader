import os
import subprocess
import argparse as ap

from utils.preprocessing import preprocessing_train
from utils.tools import printLog, printError


def parsing():
    parser = ap.ArgumentParser(
        prog='trading algo',
        description='predictive model for trading'
    )
    parser.add_argument('-EURUSD', type=str, help='EURUSD datafile')
    parser.add_argument('-lifespan', type=int, default=10, help='lifespan of the trade in days')
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


def trainModels(datafile, currency_pair):
    printLog('TRAINING GRADIENT BOOSTING MODEL...')
    subprocess.run(["python", "models/GradientBoosting.py"] + [datafile] + [currency_pair])
    printLog('GRADIENT BOOSTING MODEL DONE\n')

    printLog('TRAINING MLP MODEL...')
    subprocess.run(["python", "models/MLP.py"] + [datafile] + [currency_pair])
    printLog('MLP MODEL DONE\n')

    printLog('TRAINING LOGISTIC REGRESSION MODEL...')
    subprocess.run(["python", "models/LogReg.py"] + [datafile] + [currency_pair])
    printLog('LOGISTIC REGRESSION MODEL DONE\n')

    printLog('TRAINING RANDOM FOREST CLASSIFIER MODEL...')
    subprocess.run(["python", "models/RandomForestClassifier.py"] + [datafile] + [currency_pair])
    printLog('RANDOM FOREST CLASSIFIER MODEL DONE\n')

    printLog('TRAINING XGB MODEL...')
    subprocess.run(["python", "models/XGB.py"] + [datafile] + [currency_pair])
    printLog('XGB MODEL DONE\n')


if __name__ == '__main__':
    try:
        args = parsing()
        if args.EURUSD is not None:
#            if not os.path.exists('data/EURUSD/EURUSD_preprocessed.csv'):
            preprocessing_train('EURUSD', args, args.EURUSD)
            trainModels('data/EURUSD/EURUSD_preprocessed.csv', 'EURUSD')

    except Exception as error:
        print(error)