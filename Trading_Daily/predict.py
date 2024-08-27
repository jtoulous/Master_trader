import os
import argparse as ap
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from utils.preprocessing import preprocessing_predict
from utils.tools import printLog, printError

def parsing():
    parser = ap.ArgumentParser(
        prog='trading algo',
        description='predictive model for trading'
    )
    parser.add_argument('-BTCUSD', action='store_true', help='BTCUSD')

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
    if args.BTCUSD is not None:    
        error_check('BTCUSD')
    return args

def error_check(currency):
    if not os.path.exists(f'models/architectures/random_forest_model_{currency}.pkl')\
    or not os.path.exists(f'models/architectures/gradient_boosting_{currency}.pkl')\
    or not os.path.exists(f'models/architectures/logreg_{currency}.pkl')\
    or not os.path.exists(f'models/architectures/mlp_{currency}.pkl')\
    or not os.path.exists(f'models/architectures/xgb_{currency}.pkl')\
    or not os.path.exists(f'models/architectures/xgb_label_encoder_{currency}.pkl'):
        raise Exception(f'error: train {currency} models before making {currency} predictions')


def print_predictions(stop_loss, take_profit, predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb):
        prediction = 'Win' if predictions_rf[0] == 'W'\
                        and predictions_gb[0] == 'W'\
                        and predictions_lr[0] == 'W'\
                        and predictions_xgb[0] == 'W'\
                        and predictions_mlp[0] == 'W'\
                        else 'Lose'
        printLog(f'=========   PREDICTION   =========')
        printLog(f'====> {prediction}')
        if prediction == 'Win':    
            printLog(f'  SL ==> {stop_loss}')
            printLog(f'  TP ==> {take_profit}')


def make_predictions(dataframe, currency_pair, stop_loss, take_profit):
#    printLog('Reading data...')
    features = list(dataframe.columns)
    features.remove('DATETIME')
    features.remove('HIGH')
    features.remove('LOW')
    features.remove('CLOSE')
    features.remove('VOLUME')
    if 'LABEL' in features:
        features.remove('LABEL')
    X = dataframe[features]
#    printLog('Done')

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

    print_predictions(stop_loss, take_profit, predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb)


if __name__ == '__main__':
    try:    
        args = parsing()
        if args.BTCUSD is True:
            dataframe = pd.read_csv('data/BTCUSD/BTCUSD(D).csv')
            dataframe, stop_loss, take_profit = preprocessing_predict(args, dataframe)
            make_predictions(dataframe, 'BTCUSD', stop_loss, take_profit)
            

    except Exception as error:
        printError(error)