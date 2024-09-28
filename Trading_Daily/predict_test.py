import os
import argparse as ap
import pandas as pd
import joblib

from utils.preprocessing import preprocessing_test
from utils.log import printLog, printError, printHeader
from utils.dataframe import ReadDf
from utils.arguments import GetArg, ActiveCryptos, GetCryptoFile, UpdateArgs, GetFeatures
from utils.tools import UnanimityPrediction, SafeDivide

def parsing():
    parser = ap.ArgumentParser(
        prog='trading algo',
        description='predictive model for trading'
    )
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
    args = parser.parse_args()
    return args


def predictions_stats(y, predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb):
    total_good_pred = 0
    total_win_pred = 0
    total_lose_pred = 0

    total_true_win = 0
    total_false_win = 0
    total_true_lose = 0
    total_false_lose = 0

    for i in range(len(y)):
        prediction = UnanimityPrediction(predictions_rf[i], predictions_gb[i], predictions_lr[i], predictions_mlp[i], predictions_xgb[i])
        if prediction == 'Win':
            total_win_pred += 1
        else:
            total_lose_pred += 1

        if prediction == y[i]:
            total_good_pred += 1
            if prediction == 'Win':
                total_true_win += 1
            else:
                total_true_lose += 1
        else:
            if prediction == 'Win':
                total_false_win += 1
            else:
                total_false_lose += 1
    win_percentage = round((total_true_win / (total_true_win + total_false_win)) * 100, 2)
#    final_stats = (
#        f'\nTRUE WIN ACCURACY ===> {SafeDivide(total_true_win, total_win_pred)}%  ({total_true_win})'
#        f'\nFALSE WIN ACCURACY ===> {SafeDivide(total_false_win, total_win_pred)}%  ({total_false_win})'
#        f'\nTRUE LOSS ACCURACY ===> {SafeDivide(total_true_lose, total_lose_pred)}%  ({total_true_lose})'
#        f'\nFALSE LOSS ACCURACY ===> {SafeDivide(total_false_lose, total_lose_pred)}%  ({total_false_lose})\n'
#        f'TOTAL ACCURACY ===> {SafeDivide(total_good_pred, len(y))}% correct'
#    )
    return f'TRUE WIN ACCURACY ===> {total_true_win} / {total_false_win} ({win_percentage}%)'


def make_test_predictions(dataframe, currency_pair):
    printLog('Reading data...')
    X, y = dataframe[GetFeatures()], dataframe['LABEL']
    printLog('Done')

    printLog('Loading models...')
    random_forest = joblib.load(f'models/architectures/random_forest_model_{currency_pair}.pkl')
    gradient_boosting = joblib.load(f'models/architectures/gradient_boosting_{currency_pair}.pkl')
    logreg = joblib.load(f'models/architectures/logreg_{currency_pair}.pkl')
    mlp = joblib.load(f'models/architectures/mlp_{currency_pair}.pkl')
    xgb = joblib.load(f'models/architectures/xgb_{currency_pair}.pkl')
    label_encoder = joblib.load(f'models/architectures/xgb_label_encoder_{currency_pair}.pkl')
    printLog('Done')

    printLog('Predicting...')
    predictions_rf = random_forest.predict(X)
    predictions_gb = gradient_boosting.predict(X)
    predictions_lr = logreg.predict(X)
    predictions_mlp = mlp.predict(X)
    predictions_xgb = label_encoder.inverse_transform(xgb.predict(X))
    printLog('Done\n')

    return predictions_stats(y, predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb)
    


if __name__ == '__main__':
    try :
        args = parsing()
        pred_stats = {}

        for crypto in ActiveCryptos():
            args = UpdateArgs(args, crypto)
            printHeader(crypto)
            dataframe = ReadDf(GetCryptoFile(crypto, file_type='test predict'))
            full_dataframe = ReadDf(GetCryptoFile(crypto))
            dataframe = preprocessing_test(args, dataframe, full_dataframe)
            pred_stats[crypto] = make_test_predictions(dataframe, crypto)

        for crypto in pred_stats.keys():
            printLog(f'\n\n==============   {crypto}   ==============')
            printLog(f'{pred_stats[crypto]}')

    except Exception as error:
        printError(error)