import pandas as pd
import argparse as ap
import os

from utils.log import printError

def ArgParsing():
    parser = ap.ArgumentParser(
        prog='data cleaner',
        description='Clean raw history files'
    )
    parser.add_argument('-repo', type=str, default=None, help='raw repo to clean')
    parser.add_argument('-file', type=str, default=None, help='raw file to clean')
    return parser.parse_args()


def CleanRepo(repo):
    files = os.listdir(repo)
    for file in files:
        path = repo + '/' + file
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


def CleanFile(file):
    df = pd.read_csv(file)
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
    df_sorted.to_csv(file, index=False)


if __name__ == '__main__':
    try:
        args = ArgParsing()
        if args.repo is not None:
            CleanRepo(args.repo)
        
        if args.file is not None:
            CleanFile(args.file)

    except Exception as error:
        printError(error)