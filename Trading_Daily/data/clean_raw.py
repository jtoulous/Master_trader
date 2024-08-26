import pandas as pd
import argparse as ap

def ArgParsing():
    parser = ap.ArgumentParser(
        prog='data cleaner',
        description='Clean raw history files'
    )
    parser.add_argument('datafile', type=str, help='raw datafile to clean')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = ArgParsing()
        df = pd.read_csv(args.datafile)
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
        df_sorted.to_csv(args.datafile, index=False)

    except Exception as error:
        print(error)