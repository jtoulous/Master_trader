import os
import argparse as ap
import joblib

from models.models import GradientBoosting, LogReg, MLP, RFClassifier, XGB
#from utils.estimate import Estimator
from utils.preprocessing import preprocessing_train
from utils.log import printLog, printError, printHeader
from utils.arguments import GetArg, ActiveCryptos, GetCryptoFile, UpdateArgs

def Parsing():
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


def TrainModels(dataframe, crypto, crossval, model_type)
    printLog('\nTRAINING GRADIENT BOOSTING MODEL...')
    GradientBoosting(dataframe, currency_pair, args.crossval, 'balanced')
    printLog('GRADIENT BOOSTING MODEL DONE\n')

    printLog('TRAINING MLP MODEL...')
    MLP(dataframe, currency_pair, args.crossval, 'balanced')
    printLog('MLP MODEL DONE\n')

    printLog('TRAINING LOGISTIC REGRESSION MODEL...')
    LogReg(dataframe, currency_pair, args.crossval, 'balanced')
    printLog('LOGISTIC REGRESSION MODEL DONE\n')

    printLog('TRAINING RANDOM FOREST CLASSIFIER MODEL...')
    RFClassifier(dataframe, currency_pair, args.crossval, 'balanced')
    printLog('RANDOM FOREST CLASSIFIER MODEL DONE\n')

    printLog('TRAINING XGB MODEL...')
    XGB(dataframe, currency_pair, args.crossval, 'balanced')
    printLog('XGB MODEL DONE\n')


def BalancedModels(dataframe, crypto, args):
    preprocess_pipeline = Pipeline([
        ('indicators_calculator', IndicatorCalculator(args)),
        ('label_calculator', LabelCalculator(args)),
        ('cleaner', Cleaner()),
        ('sampler', BalancedOverSampler()),
        ('scaler',  StandardScaler())
    ])
    dataframe = preprocess_pipeline.fit_transform(dataframe)
    TrainModels(dataframe, crypto, args, 'balanced')
    joblib.dump(scaler, f'models/architectures/scaler_{currency_pair}_balanced.joblib')


def WinModels(dataframe, crypto, args):
    preprocess_pipeline = Pipeline([
        ('indicators_calculator', IndicatorCalculator(args)),
        ('label_calculator', LabelCalculator(args)),
        ('cleaner', Cleaner()),
        ('sampler', UnbalancedSampler("win")),
        ('scaler',  StandardScaler())
    ])
    dataframe = preprocess_pipeline.fit_transform(dataframe)
    TrainModels(dataframe, crypto, args, 'win')
    joblib.dump(scaler, f'models/architectures/scaler_{currency_pair}_win.joblib')


def LoseModels(dataframe, crypto, args):
    preprocess_pipeline = Pipeline([
        ('indicators_calculator', IndicatorCalculator(args)),
        ('label_calculator', LabelCalculator(args)),
        ('cleaner', Cleaner()),
        ('sampler', UnbalancedSampler("lose")),
        ('scaler',  StandardScaler())
    ])
    dataframe = preprocess_pipeline.fit_transform(dataframe)
    TrainModels(dataframe, crypto, args, 'lose')
    joblib.dump(scaler, f'models/architectures/scaler_{currency_pair}_lose.joblib')


if __name__ == '__main__':
    try:
        args = Parsing()

        for crypto in ActiveCryptos():
            printHeader(f'{crypto}')
            args = UpdateArgs(args, crypto)
            file = GetCryptoFile(crypto) if args.test is False else GetCryptoFile(crypto, file_type='test train')
            dataframe = ReadDf(file)

            BalancedModels(dataframe.copy(), crypto, args)  # model trained with balanced data (50% Win, 50% Lose)
            WinModels(dataframe.copy(), crypto, args)   # model specialized in predicting wins by unbalancing the training data so that there are more winning cases in the data (66% Win, 33% Lose)
            LoseModels(dataframe.copy(), crypto, args)  # models specialized in predicting losses (33% Win, 66% Lose)


    except Exception as error:
        printError(error)