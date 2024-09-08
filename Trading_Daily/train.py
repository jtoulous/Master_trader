import os
import argparse as ap
import joblib

from models.models import GradientBoosting, LogReg, MLP, RFClassifier, XGB
from utils.preprocessing import preprocessing_train
from utils.log import printLog, printError, printHeader
from utils.arguments import GetArg, ActiveCryptos, GetCryptoFile, UpdateArgs

def parsing():
    parser = ap.ArgumentParser(
        prog='trading algo',
        description='predictive model for trading'
    )
    parser.add_argument('-crossval', action='store_true', help='run cross validation at training')
    parser.add_argument('-test', action='store_true', help='run training on test_train.csv instead')

    parser.add_argument('-lifespan', type=int, default=GetArg('lifespan'), help='lifespan of the trade in days')
    parser.add_argument('-risk', type=float, default=0, help='percentage of capital for the stop-loss')
    parser.add_argument('-profit', type=float, default=0, help='percentage of capital for the take-profit')
    parser.add_argument('-atr', type=int, default=GetArg('atr'), help='periods used for calculating ATR')
    parser.add_argument('-ema', type=int, default=GetArg('ema'), help='periods used for calculating EMA')
    parser.add_argument('-rsi', type=int, default=GetArg('rsi'), help='periods used for calculating RSI')
    parser.add_argument('-sto', type=int, default=GetArg('sto'), nargs=2, help='periods used for calculating STO')
    parser.add_argument('-sma', type=int, default=GetArg('sma'), help='periods used for calculating SMA')
    parser.add_argument('-wma', type=int, default=GetArg('wma'), help='periods used for calculating WMA')
    parser.add_argument('-dmi', type=int, default=GetArg('dmi'), help='periods used for calculating DMI')
    parser.add_argument('-blg', type=int, default=GetArg('blg'), nargs=2, help='periods and nb_stddev used for calculating Bollingers bands')
    parser.add_argument('-macd', type=int, default=GetArg('macd'), nargs=3, help='periods(short, long, signal) used for calculating MACD')
    parser.add_argument('-cci', type=int, default=GetArg('cci'), help='periods used for calculating CCI')
    parser.add_argument('-ppo', type=int, default=GetArg('ppo'), nargs=3, help='periods(short, long, signal) used for calculating PPO')
    return parser.parse_args()


def trainModels(dataframe, currency_pair, scaler, crossval):
    printLog('\nTRAINING GRADIENT BOOSTING MODEL...')
    GradientBoosting(dataframe, currency_pair, crossval)
    printLog('GRADIENT BOOSTING MODEL DONE\n')

    printLog('TRAINING MLP MODEL...')
    MLP(dataframe, currency_pair, crossval)
    printLog('MLP MODEL DONE\n')

    printLog('TRAINING LOGISTIC REGRESSION MODEL...')
    LogReg(dataframe, currency_pair, crossval)
    printLog('LOGISTIC REGRESSION MODEL DONE\n')

    printLog('TRAINING RANDOM FOREST CLASSIFIER MODEL...')
    RFClassifier(dataframe, currency_pair, crossval)
    printLog('RANDOM FOREST CLASSIFIER MODEL DONE\n')

    printLog('TRAINING XGB MODEL...')
    XGB(dataframe, currency_pair, crossval)
    printLog('XGB MODEL DONE\n')

    joblib.dump(scaler, f'models/architectures/scaler_{currency_pair}.joblib')


if __name__ == '__main__':
    try:
        args = parsing()

        for crypto in ActiveCryptos():
            args = UpdateArgs(args, crypto)
            printHeader(f'{crypto}')
            file = GetCryptoFile(crypto) if args.test is False else GetCryptoFile(crypto, file_type='test train')
            dataframe, scaler = preprocessing_train(args, file)
            trainModels(dataframe, crypto, scaler, args.crossval)


    except Exception as error:
        printError(error)