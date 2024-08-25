import os
import argparse as ap
import pandas as pd
import joblib

from utils.preprocessing import preprocessing_predict
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
    args = parser.parse_args()
    error_check(args)
    return args

def error_check(args):
    if args.EURUSD is not None:
        if not os.path.exists('models/random_forest_model_EURUSD.pkl')\
        or not os.path.exists('models/gradient_boosting_EURUSD.pkl')\
        or not os.path.exists('models/logreg_EURUSD.pkl')\
        or not os.path.exists('models/mlp_EURUSD.pkl')\
        or not os.path.exists('models/xgb_EURUSD.pkl')\
        or not os.path.exists('models/xgb_label_encoder_EURUSD.pkl'):
            raise Exception('error: train EURUSD models before making EURUSD predictions')


def print_predictions(dataframe, predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb):
    for idx, row in dataframe.iterrows():
        prediction = 'W' if predictions_rf[idx] == 'W'\
                        and predictions_gb[idx] == 'W'\
                        and predictions_lr[idx] == 'W'\
                        and predictions_xgb[idx] == 'W'\
                        and predictions_mlp[idx] == 'W'\
                        else 'L'
        printLog(f'{row["DATETIME"]} ===> {prediction}')


def make_predictions(dataframe, currency_pair):
    printLog('Reading data...')
    features = list(dataframe.columns)
    features.remove('DATETIME')
    if 'LABEL' in features:
        features.remove('LABEL')
    X = dataframe[features]
    printLog('Done')

    printLog('Loading models...')
    random_forest = joblib.load(f'models/random_forest_model_{currency_pair}.pkl')
    gradient_boosting = joblib.load(f'models/gradient_boosting_{currency_pair}.pkl')
    logreg = joblib.load(f'models/logreg_{currency_pair}.pkl')
    mlp = joblib.load(f'models/mlp_{currency_pair}.pkl')
    xgb = joblib.load(f'models/xgb_{currency_pair}.pkl')
    label_encoder = joblib.load(f'models/xgb_label_encoder_{currency_pair}.pkl')
    printLog('Done')

    printLog('Predicting...')
    predictions_rf = random_forest.predict(X)
    predictions_gb = gradient_boosting.predict(X)
    predictions_lr = logreg.predict(X)
    predictions_mlp = mlp.predict(X)
    predictions_xgb = label_encoder.inverse_transform(xgb.predict(X))
    printLog('Done\n')

    print_predictions(dataframe, predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb)


if __name__ == '__main__':
    try:    
        args = parsing()
        if args.EURUSD is not None:
            dataframe = pd.read_csv(args.EURUSD, index_col=False)
#            if len(dataframe.columns) < 30:
            dataframe = preprocessing_predict(args, dataframe)
            make_predictions(dataframe, 'EURUSD')

    except Exception as error:
        printError(error)