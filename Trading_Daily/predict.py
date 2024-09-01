import os
import argparse as ap
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from utils.preprocessing import preprocessing_predict
from utils.log import printLog, printError
from utils.dataframe import ReadDf
from utils.arguments import GetArg

def parsing():
    parser = ap.ArgumentParser(
        prog='trading algo',
        description='predictive model for trading'
    )
    parser.add_argument('-BTC', type=str, default='data/BTC-USD/BTC-USD.csv', help='BTC-USD datafile')
    parser.add_argument('-ETH', type=str, default='data/ETH-USD/ETH-USD.csv', help='ETH-USD datafile')
    parser.add_argument('-BNB', type=str, default='data/BNB-USD/BNB-USD.csv', help='BNB-USD datafile')
    parser.add_argument('-SOL', type=str, default='data/SOL-USD/SOL-USD.csv', help='SOL-USD datafile')
    parser.add_argument('-ADA', type=str, default='data/ADA-USD/ADA-USD.csv', help='ADA-USD datafile')
    parser.add_argument('-LINK', type=str, default='data/LINK-EUR/LINK-EUR.csv', help='LINKEUR datafile')
    parser.add_argument('-AVAX', type=str, default='data/AVAX-EUR/AVAX-EUR.csv', help='AVAXEUR datafile')
    parser.add_argument('-DOT', type=str, default='data/DOT-JPY/DOT-JPY.csv', help='DOTJPY datafile')

    parser.add_argument('-date', type=str, default=None, help='prediction date')
    parser.add_argument('-risk', type=float, default=GetArg('risk'), help='percentage of capital for the stop-loss')
    parser.add_argument('-profit', type=float, default=GetArg('profit'), help='percentage of capital for the take-profit')
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
    
    error_check('BTC-USD')
    error_check('ETH-USD')
    error_check('SOL-USD')
    error_check('BNB-USD')
    error_check('ADA-USD')
    return args

def error_check(currency):
    if not os.path.exists(f'models/architectures/random_forest_model_{currency}.pkl')\
    or not os.path.exists(f'models/architectures/gradient_boosting_{currency}.pkl')\
    or not os.path.exists(f'models/architectures/logreg_{currency}.pkl')\
    or not os.path.exists(f'models/architectures/mlp_{currency}.pkl')\
    or not os.path.exists(f'models/architectures/xgb_{currency}.pkl')\
    or not os.path.exists(f'models/architectures/xgb_label_encoder_{currency}.pkl'):
        raise Exception(f'error: train {currency} models before making {currency} predictions')


def print_predictions(currency, stop_loss, take_profit, open, predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb):
        prediction = 'Win' if predictions_rf[0] == 'W'\
                        and predictions_gb[0] == 'W'\
                        and predictions_lr[0] == 'W'\
                        and predictions_xgb[0] == 'W'\
                        and predictions_mlp[0] == 'W'\
                        else 'Lose'
        printLog(f'\n=========   PREDICTION {currency}  =========')
        printLog(f'======> {prediction}')  
        printLog(f'  OPEN == {open}')
        printLog(f'  SL == {stop_loss}')
        printLog(f'  TP == {take_profit}/n')


def make_predictions(dataframe, currency_pair, stop_loss, take_profit, open):
    features = list(dataframe.columns)
    features.remove('DATETIME')
    features.remove('HIGH')
    features.remove('LOW')
    features.remove('CLOSE')
    features.remove('VOLUME')
    if 'LABEL' in features:
        features.remove('LABEL')
#    X = dataframe[features]

    random_forest = joblib.load(f'models/architectures/random_forest_model_{currency_pair}.pkl')
    gradient_boosting = joblib.load(f'models/architectures/gradient_boosting_{currency_pair}.pkl')
    logreg = joblib.load(f'models/architectures/logreg_{currency_pair}.pkl')
    mlp = joblib.load(f'models/architectures/mlp_{currency_pair}.pkl')
    xgb = joblib.load(f'models/architectures/xgb_{currency_pair}.pkl')
    label_encoder = joblib.load(f'models/architectures/xgb_label_encoder_{currency_pair}.pkl')
    scaler = joblib.load(f'models/architectures/scaler_{currency_pair}.joblib')
    
    features_df = dataframe[features]
    scaled_features = scaler.transform(features_df)
    scaled_features_df =  pd.DataFrame(scaled_features, columns=features)
    dataframe[features] = scaled_features_df
#    breakpoint()

    X = dataframe[features]
    predictions_rf = random_forest.predict(X)
    predictions_gb = gradient_boosting.predict(X)
    predictions_lr = logreg.predict(X)
    predictions_mlp = mlp.predict(X)
    predictions_xgb = label_encoder.inverse_transform(xgb.predict(X))

    print_predictions(currency_pair, stop_loss, take_profit, open, predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb)


if __name__ == '__main__':  ####FAIRE UN AUTO UPDATE AVANT DE COMMENCER
    try:    
        args = parsing()
        
        dataframe = ReadDf(args.BTC)
        dataframe, stop_loss, take_profit, open = preprocessing_predict(args, dataframe, 'BTC-USD')
        make_predictions(dataframe, 'BTC-USD', stop_loss, take_profit, open)
            
        dataframe = ReadDf(args.ETH)
        dataframe, stop_loss, take_profit, open = preprocessing_predict(args, dataframe, 'ETH-USD')
        make_predictions(dataframe, 'ETH-USD', stop_loss, take_profit, open)

        dataframe = ReadDf(args.BNB)
        dataframe, stop_loss, take_profit, open = preprocessing_predict(args, dataframe, 'BNB-USD')
        make_predictions(dataframe, 'BNB-USD', stop_loss, take_profit, open)

        dataframe = ReadDf(args.SOL)
        dataframe, stop_loss, take_profit, open = preprocessing_predict(args, dataframe, 'SOL-USD')
        make_predictions(dataframe, 'SOL-USD', stop_loss, take_profit, open)

        dataframe = ReadDf(args.ADA)
        dataframe, stop_loss, take_profit, open = preprocessing_predict(args, dataframe, 'ADA-USD')
        make_predictions(dataframe, 'ADA-USD', stop_loss, take_profit, open)

        dataframe = ReadDf(args.LINK)
        dataframe, stop_loss, take_profit, open = preprocessing_predict(args, dataframe, 'LINK-EUR')
        make_predictions(dataframe, 'LINK-EUR', stop_loss, take_profit, open)

        dataframe = ReadDf(args.AVAX)
        dataframe, stop_loss, take_profit, open = preprocessing_predict(args, dataframe, 'AVAX-EUR')
        make_predictions(dataframe, 'AVAX-EUR', stop_loss, take_profit, open)

        dataframe = ReadDf(args.DOT)
        dataframe, stop_loss, take_profit, open = preprocessing_predict(args, dataframe, 'DOT-JPY')
        make_predictions(dataframe, 'DOT-JPY', stop_loss, take_profit, open)
 
    except Exception as error:
        printError(error)