import pandas as pd

def ReadDf(csv_file):
    df = pd.read_csv(csv_file)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.sort_values(by='DATETIME', inplace=True)
    return df

def CleanDf(df):
    df.reset_index(inplace=True)
    df = df.drop(columns=['Adj Close'])
    df.rename(columns={
        'Date': 'DATETIME',
        'Open': 'OPEN',
        'High': 'HIGH',
        'Low': 'LOW',
        'Close': 'CLOSE',
        'Volume': 'VOLUME'
    }, inplace=True)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    return df