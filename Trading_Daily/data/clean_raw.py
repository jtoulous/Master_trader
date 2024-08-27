import pandas as pd
import argparse as ap
import os

def ArgParsing():
    parser = ap.ArgumentParser(
        prog='data cleaner',
        description='Clean raw history files'
    )
    parser.add_argument('datarepo', type=str, help='raw datarepo to clean')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = ArgParsing()
        files = os.listdir(args.datarepo)
        
        for file in files:
            path = args.datarepo + '/' + file
            df = pd.read_csv(path)
            df.rename(columns={
                'Date': 'DATETIME',
                'Open': 'OPEN',
                'High': 'HIGH',
                'Low': 'LOW',
                'Close': 'CLOSE',
                'Volume': 'VOLUME'
            }, inplace=True)
        
            df = df.drop(columns=['Adj Close'])
            df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            df_sorted = df.sort_values(by='DATETIME')
            df_sorted.to_csv(path, index=False)

    except Exception as error:
        print(error)