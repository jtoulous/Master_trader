import os
import argparse as ap
import pandas as pd
import joblib

from utils.preprocessing import preprocessing_test
from utils.tools import printLog, printError

def parsing():
    parser = ap.ArgumentParser(
        prog='trading algo',
        description='predictive model for trading'
    )
    parser.add_argument('-EURUSD', type=str, default=None, help='EURUSD datafile')
    parser.add_argument('-GBPUSD', type=str, default=None, help='GBPUSD datafile')

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
    args = parser.parse_args()
    if args.EURUSD is not None:    
        error_check('EURUSD')
    if args.GBPUSD is not None:    
        error_check('GBPUSD')
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
#    printLog(f'\nTRUE WIN ACCURACY ===> {(total_true_win / total_win_pred) * 100}%  ({total_true_win})')
#    printLog(f'FALSE WIN ACCURACY ===> {(total_false_win / total_win_pred) * 100}%  ({total_false_win})\n')
#    printLog(f'TRUE LOSS ACCURACY ===> {(total_true_lose / total_lose_pred) * 100}%  ({total_true_lose})')
#    printLog(f'FALSE LOSS ACCURACY ===> {(total_false_lose / total_lose_pred) * 100}%  ({total_false_lose})\n')        
#    printLog(f'TOTAL ACCURACY ===> {(total_good_pred / len(y)) * 100}% correct')


def make_test_predictions(dataframe, currency_pair):
    printLog('Reading data...')
    features = list(dataframe.columns)
    features.remove('LABEL')
    features.remove('DATETIME')
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
        if args.EURUSD is not None:
            printLog('\n=============================================================')
            printLog('||                          EURUSD                          ||')
            printLog('=============================================================')
            dataframe = pd.read_csv(args.EURUSD, index_col=False)
            dataframe = preprocessing_test(args, dataframe)
            pred_stats['EURUSD'] = make_test_predictions(dataframe, 'EURUSD')

        if args.GBPUSD is not None:
            printLog('\n=============================================================')
            printLog('||                          GBPUSD                          ||')
            printLog('=============================================================')
            dataframe = pd.read_csv(args.GBPUSD, index_col=False)
            dataframe = preprocessing_test(args, dataframe)
            pred_stats['GBPUSD'] = make_test_predictions(dataframe, 'EURUSD')

        for currency in pred_stats.keys():
            printLog(f'\n\n==============   {currency}   ==============')
            printLog(f'{pred_stats[currency]}')

    except Exception as error:
        printError(error)