import os
import argparse as ap
import pandas as pd
import joblib

from utils.preprocessing import preprocessing_test
from utils.log import printLog, printError, printHeader
from utils.dataframe import ReadDf

def parsing():
    parser = ap.ArgumentParser(
        prog='trading algo',
        description='predictive model for trading'
    )
    parser.add_argument('-BTC', type=str, default='data/BTC-USD/test_predict.csv', help='BTCUSD datafile')
    parser.add_argument('-ETH', type=str, default='data/ETH-USD/test_predict.csv', help='ETHUSD datafile')
    parser.add_argument('-BNB', type=str, default='data/BNB-USD/test_predict.csv', help='BNBUSD datafile')
    parser.add_argument('-SOL', type=str, default='data/SOL-USD/test_predict.csv', help='SOLUSD datafile')
    parser.add_argument('-ADA', type=str, default='data/ADA-USD/test_predict.csv', help='ADAUSD datafile')

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
    args = parser.parse_args()
    error_check('BTC-USD')
    error_check('ETH-USD')
    error_check('BNB-USD')
    return args

def error_check(currency_pair):
    if not os.path.exists(f'models/architectures/random_forest_model_{currency_pair}.pkl')\
    or not os.path.exists(f'models/architectures/gradient_boosting_{currency_pair}.pkl')\
    or not os.path.exists(f'models/architectures/logreg_{currency_pair}.pkl')\
    or not os.path.exists(f'models/architectures/mlp_{currency_pair}.pkl')\
    or not os.path.exists(f'models/architectures/xgb_{currency_pair}.pkl')\
    or not os.path.exists(f'models/architectures/xgb_label_encoder_{currency_pair}.pkl'):
        raise Exception(f'error: train {currency_pair} models before making {currency_pair} predictions')


def predictions_stats(y, predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb):
    total_good_pred = 0
    total_win_pred = 0
    total_lose_pred = 0

    total_true_win = 0
    total_false_win = 0
    total_true_lose = 0
    total_false_lose = 0
    for i in range(len(y)):
        prediction = 'W' if predictions_rf[i] == 'W'\
                        and predictions_gb[i] == 'W'\
                        and predictions_lr[i] == 'W'\
                        and predictions_xgb[i] == 'W'\
                        and predictions_mlp[i] == 'W'\
                        else 'L'
        if prediction == 'W':
            total_win_pred += 1
        else:
            total_lose_pred += 1

        if prediction == y[i]:
            total_good_pred += 1
            if prediction == 'W':
                total_true_win += 1
            else:
                total_true_lose += 1
            printLog(f'{y[i]} ===> {prediction}')

        else:
            if prediction == 'W':
                total_false_win += 1
            else:
                total_false_lose += 1
            printError(f'{y[i]} ===> {prediction}')

    final_stats = (
            f'\nTRUE WIN ACCURACY ===> {(total_true_win / total_win_pred) * 100}%  ({total_true_win})'
            f'\nFALSE WIN ACCURACY ===> {(total_false_win / total_win_pred) * 100}%  ({total_false_win})'
            f'\nTRUE LOSS ACCURACY ===> {(total_true_lose / total_lose_pred) * 100}%  ({total_true_lose})'
            f'\nFALSE LOSS ACCURACY ===> {(total_false_lose / total_lose_pred) * 100}%  ({total_false_lose})\n'
            f'TOTAL ACCURACY ===> {(total_good_pred / len(y)) * 100}% correct'
    )
    return final_stats


def make_test_predictions(dataframe, currency_pair):
    printLog('Reading data...')
    features = list(dataframe.columns)
    features.remove('LABEL')
    features.remove('DATETIME')
    features.remove('HIGH')
    features.remove('LOW')
    features.remove('CLOSE')
    features.remove('VOLUME')
    X = dataframe[features]
    y = dataframe['LABEL']
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

        printHeader('BTCUSD')
        dataframe = ReadDf(args.BTC)
        dataframe = preprocessing_test(args, dataframe)
        pred_stats['BTCUSD'] = make_test_predictions(dataframe, 'BTC-USD')

        printHeader('ETHUSD')
        dataframe = ReadDf(args.ETH)
        dataframe = preprocessing_test(args, dataframe)
        pred_stats['ETHUSD'] = make_test_predictions(dataframe, 'ETH-USD')

        printHeader('BNBUSD')
        dataframe = ReadDf(args.BNB)
        dataframe = preprocessing_test(args, dataframe)
        pred_stats['BNBUSD'] = make_test_predictions(dataframe, 'BNB-USD')

        printHeader('SOLUSD')
        dataframe = ReadDf(args.SOL)
        dataframe = preprocessing_test(args, dataframe)
        pred_stats['SOLUSD'] = make_test_predictions(dataframe, 'SOL-USD')

        printHeader('ADAUSD')
        dataframe = ReadDf(args.ADA)
        dataframe = preprocessing_test(args, dataframe)
        pred_stats['ADAUSD'] = make_test_predictions(dataframe, 'ADA-USD')


        for currency in pred_stats.keys():
            printLog(f'\n\n==============   {currency}   ==============')
            printLog(f'{pred_stats[currency]}')

    except Exception as error:
        printError(error)