import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from .tools import printLog, printInfo, printError
from .indicators import calc_indicators


def calc_labels(dataframe, args):
    printLog('Defining labels...')
    lifespan = args.lifespan
    risk = args.risk
    profit = args.profit
    dataframe = dataframe.assign(LABEL=None)
    
    for r, row in dataframe.iterrows():
        datetime_range = pd.date_range(
            start=row['DATETIME'] + pd.Timedelta(minutes=1),
            end=row['DATETIME'] + pd.Timedelta(minutes=args.lifespan),
            freq='1min'
        )
        take_profit = row['OPEN'] + (row['ATR'] * args.profit )
        stop_loss = row['OPEN'] - (row['ATR'] * args.risk)
        for datetime in datetime_range:
            idx = dataframe['DATETIME'].searchsorted(datetime)
        
            if idx < len(dataframe) and dataframe.loc[idx, 'DATETIME'] == datetime:
                high = dataframe.iloc[idx, dataframe.columns.get_loc('HIGH')]
                low = dataframe.iloc[idx, dataframe.columns.get_loc('LOW')]
                if high >= take_profit:
                    print(f'{row["DATETIME"]} ==> win')
                    dataframe.at[r, "LABEL"] = 'W'
                    break
        
                if low <= stop_loss or datetime > datetime_range[-2]:
                    print(f'{row["DATETIME"]} ==> loss')
                    dataframe.at[r, "LABEL"] = 'L'
                    break
            else:
                    print(f'{row["DATETIME"]} ==> loss')
                    dataframe.at[r, "LABEL"] = 'L'
                    break
    dataframe.insert(1, 'LABEL', dataframe.pop('LABEL'))
    printLog('Done')
    return dataframe


def preprocessing_train(currency_pair, args, datafile):
    dataframe = pd.read_csv(datafile, index_col=False)
    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])
    dataframe = dataframe.sort_values(by='DATETIME')
    dataframe  = calc_indicators(dataframe, args) 
    dataframe = calc_labels(dataframe, args)
    dataframe = dataframe.drop(dataframe.index[:200])
    dataframe.reset_index(drop=True, inplace=True)
    dataframe.bfill(inplace=True)

    scaler = StandardScaler()
    features = list(dataframe.columns)
    features.remove('LABEL')
    features.remove('DATETIME')
    features_df = dataframe[features]
    scaled_features = scaler.fit_transform(features_df)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features)
    dataframe[features] = scaled_features_df
    dataframe.to_csv(f'data/{currency_pair}/{currency_pair}_preprocessed.csv', index=False)


def preprocessing_test(args, dataframe):
    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])
    dataframe = dataframe.sort_values(by='DATETIME')
    dataframe  = calc_indicators(dataframe, args) 
    dataframe = calc_labels(dataframe, args)
    dataframe.bfill(inplace=True)

    scaler = StandardScaler()
    features = list(dataframe.columns)
    features.remove('LABEL')
    features.remove('DATETIME')
    features_df = dataframe[features]
    scaled_features = scaler.fit_transform(features_df)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features)
    dataframe[features] = scaled_features_df
    return dataframe


def preprocessing_predict(args, dataframe):
    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])
    dataframe = dataframe.sort_values(by='DATETIME')
    dataframe  = calc_indicators(dataframe, args) 
    dataframe.bfill(inplace=True)

    scaler = StandardScaler()
    features = list(dataframe.columns)
    features.remove('DATETIME')
    features_df = dataframe[features]
    scaled_features = scaler.fit_transform(features_df)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features)
    dataframe[features] = scaled_features_df
    return dataframe