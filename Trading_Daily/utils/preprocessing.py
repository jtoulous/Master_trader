import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from .tools import printLog, printInfo, printError
from .indicators import calc_indicators


def calc_labels(dataframe, args):
    printLog('Defining labels...')
    dataframe = dataframe.assign(LABEL=None)  
    for r, row in dataframe.iterrows():
        datetime_range = pd.date_range(
            start=row['DATETIME'] + pd.Timedelta(days=1),
            end=row['DATETIME'] + pd.Timedelta(days=args.lifespan),
            freq='1D'
        )
        take_profit = row['OPEN'] + (row['ATR'] * args.profit)
        stop_loss = row['OPEN'] - (row['ATR'] * args.risk)
        label_assigned = False

        for datetime in datetime_range:
            idx = dataframe['DATETIME'].searchsorted(datetime)

            if idx < len(dataframe) and dataframe.loc[idx, 'DATETIME'] == datetime:
                high = dataframe.iloc[idx, dataframe.columns.get_loc('HIGH')]
                low = dataframe.iloc[idx, dataframe.columns.get_loc('LOW')]

                if low <= stop_loss:
                    dataframe.at[r, "LABEL"] = 'L'
                    label_assigned = True
                    break  

                if high >= take_profit:
                    dataframe.at[r, "LABEL"] = 'W'
                    label_assigned = True
                    break  

        if not label_assigned:
            dataframe.at[r, "LABEL"] = 'L'
    dataframe.insert(1, 'LABEL', dataframe.pop('LABEL'))
    printLog('Done')
    return dataframe


def preprocessing_train(currency_pair, args, datafile):
    dataframe = pd.read_csv(datafile, index_col=False)
    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])
    dataframe = dataframe.sort_values(by='DATETIME')
    dataframe  = calc_indicators(dataframe, args) 
    dataframe = calc_labels(dataframe, args)
    dataframe = dataframe.drop(dataframe.index[:10])
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
    return dataframe

def preprocessing_test(args, dataframe):
    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])
    dataframe = dataframe.sort_values(by='DATETIME')
    dataframe  = calc_indicators(dataframe, args) 
    dataframe = calc_labels(dataframe, args)
    dataframe = dataframe.drop(dataframe.index[:10])
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