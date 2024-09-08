import pandas as pd

from sklearn.preprocessing import StandardScaler
from .log import printLog
from .indicators import calc_indicators
from .dataframe import ReadDf
from .estimate import Estimate
from .arguments import GetOpen, GetDate

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


def preprocessing_train(args, datafile):
    dataframe = ReadDf(datafile)
    dataframe  = calc_indicators(dataframe, args) 
    dataframe = calc_labels(dataframe, args)
    dataframe = dataframe.drop(dataframe.index[:10])
    dataframe.reset_index(drop=True, inplace=True)
    dataframe.bfill(inplace=True)

    scaler = StandardScaler()
    features = list(dataframe.columns)
    features.remove('LABEL')
    features.remove('DATETIME')
    features.remove('HIGH')
    features.remove('LOW')
    features.remove('CLOSE')
    features.remove('VOLUME')
    features_df = dataframe[features]
    scaled_features = scaler.fit_transform(features_df)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features)
    dataframe[features] = scaled_features_df
    return dataframe, scaler

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


def preprocessing_predict(args, dataframe, crypto):
    if args.old is False:
        new_row = pd.DataFrame({
            'DATETIME': GetDate(args, dataframe),
            'OPEN': GetOpen(args, dataframe, crypto),
            'HIGH': [Estimate(dataframe, args, 'HIGH')],
            'LOW': [Estimate(dataframe, args, 'LOW')],
            'CLOSE': [Estimate(dataframe, args, 'CLOSE')],
            'VOLUME': [None],
        })
        dataframe = pd.concat([dataframe, new_row], ignore_index=True)
        dataframe  = calc_indicators(dataframe, args)
        take_profit = dataframe.iloc[-1]['OPEN'] + (args.profit * dataframe.iloc[-1]['ATR'])
        stop_loss = dataframe.iloc[-1]['OPEN'] - (args.risk * dataframe.iloc[-1]['ATR'])
        open = dataframe.iloc[-1]['OPEN']
        dataframe = dataframe.tail(1).reset_index(drop=True)

    else:
        date = pd.to_datetime(args.date, format='%d/%m/%Y')
        dataframe  = calc_indicators(dataframe, args)
        dataframe = dataframe[dataframe['DATETIME'] == date]
        take_profit = dataframe.iloc[-1]['OPEN'] + (args.profit * dataframe.iloc[-1]['ATR'])
        stop_loss = dataframe.iloc[-1]['OPEN'] - (args.risk * dataframe.iloc[-1]['ATR'])
        open = dataframe.iloc[-1]['OPEN']

    return dataframe, stop_loss, take_profit, open