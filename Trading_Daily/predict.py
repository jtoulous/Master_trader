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
        if args.BTCUSD is True:
            dataframe = pd.read_csv('data/BTCUSD/BTCUSD(D).csv')
            new_row = pd.DataFrame({
                'DATETIME': [None],
                'OPEN': [None],
                'HIGH': [None],
                'LOW': [None],
                'CLOSE': [None],
                'VOLUME': [None]
            })
            dataframe = pd.concat([dataframe, new_row], ignore_index=True)
            dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])
            dataframe = dataframe.sort_values(by='DATETIME')
            
            dataframe.at[dataframe.index[-1], 'OPEN'] = dataframe.iloc[-2]['CLOSE']
            dataframe.at[dataframe.index[-1], 'DATETIME'] = dataframe.iloc[-2]['DATETIME'] + pd.DateOffset(days=1)
            dataframe = preprocessing_predict(args, dataframe)
            breakpoint()
#            day_open = float(input('Open: '))

#            new_row = pd.DataFrame({
#                'DATETIME': [None],
#                'OPEN': [day_open],
#                'HIGH': [None],
#                'LOW': [None],
#                'CLOSE': [None],
#                'ATR': [None],
#                'EMA': [None],
#                'RSI': [None],
#                'MACD_LINE': [None],
#                'MACD_SIGNAL': [None],
#                'SMA': [None],
#                'WMA': [None],
#                'Hilbert_Transform': [None],
#                'PPO_LINE': [None],
#                'PPO_SIGNAL': [None],
#                'PPO_LINE': [None],
#                'ROC': [None]
#            })
            
            

    except Exception as error:
        printError(error)




##     FAIRE LES FONCTIONS POUR CALCULER LES INDICATEURS DE LA DERNIERE ENTREE SEULEMENT
