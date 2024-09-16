import os
import argparse as ap
import pandas as pd
import joblib
import datetime

from sklearn.preprocessing import StandardScaler
from utils.preprocessing import preprocessing_predict
from utils.log import printLog, printError
from utils.dataframe import ReadDf
from utils.arguments import GetArg, ActiveCryptos, GetCryptoFile, UpdateArgs, GetFeatures
from utils.tools import UnanimityPrediction

def parsing():
    parser = ap.ArgumentParser(
        prog='trading algo',
        description='predictive model for trading'
    )
    parser.add_argument('-crypto', type=str, default=None, help='crypto to predict')
    parser.add_argument('-old', action='store_true', help='old date')
    parser.add_argument('-date', type=str, default=datetime.datetime.today().strftime('%d/%m/%Y'), help='prediction date')
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

def print_predictions(currency, stop_loss, take_profit, open_pos, predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb):
        prediction = UnanimityPrediction(predictions_rf[0], predictions_gb[0], predictions_lr[0], predictions_xgb[0], predictions_mlp[0])
        if prediction == 'Win':   
            printLog(f'=========   PREDICTION {currency}  =========')
            printLog(f'======> {prediction}')  
            printLog(f'  OPEN == {open_pos}  |  SL == {stop_loss}  |    TP == {take_profit}\n')
        else:
            printError(f'=========   PREDICTION {currency}  =========')
            printError(f'======> {prediction}')  
            printError(f'  OPEN == {open_pos}  |  SL == {stop_loss}  |    TP == {take_profit}\n')


def make_predictions(dataframe, currency_pair, stop_loss, take_profit, open_pos):
    features = GetFeatures()

    random_forest = joblib.load(f'models/architectures/random_forest_model_{currency_pair}.pkl')
    gradient_boosting = joblib.load(f'models/architectures/gradient_boosting_{currency_pair}.pkl')
    logreg = joblib.load(f'models/architectures/logreg_{currency_pair}.pkl')
    mlp = joblib.load(f'models/architectures/mlp_{currency_pair}.pkl')
    xgb = joblib.load(f'models/architectures/xgb_{currency_pair}.pkl')
    label_encoder = joblib.load(f'models/architectures/xgb_label_encoder_{currency_pair}.pkl')
    scaler = joblib.load(f'models/architectures/scaler_{currency_pair}.joblib')
    
    features_df = dataframe[features]
    scaled_features = scaler.transform(features_df)
    X = pd.DataFrame(scaled_features, columns=features)
   
    predictions_rf = random_forest.predict(X)
    predictions_gb = gradient_boosting.predict(X)
    predictions_lr = logreg.predict(X)
    predictions_mlp = mlp.predict(X)
    predictions_xgb = label_encoder.inverse_transform(xgb.predict(X))

    print_predictions(currency_pair, stop_loss, take_profit, open_pos, predictions_rf, predictions_gb, predictions_lr, predictions_mlp, predictions_xgb)


if __name__ == '__main__':
    try:    
        args = parsing()
        crypto_list = [args.crypto] if args.crypto is not None else ActiveCryptos()
        printLog(f'=======>  {args.date}\n\n')
        for crypto in crypto_list:
            args = UpdateArgs(args, crypto)
            dataframe = ReadDf(GetCryptoFile(crypto))
            dataframe, stop_loss, take_profit, open_pos = preprocessing_predict(args, dataframe, crypto)
            make_predictions(dataframe, crypto, stop_loss, take_profit, open_pos)
 
    except Exception as error:
        printError(error)