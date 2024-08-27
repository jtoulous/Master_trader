import pandas as pd
import argparse as ap


def ArgParsing():
    parser = ap.ArgumentParser()
    parser.add_argument('-file', type=str, required=True, help='csv file to split')
    parser.add_argument('-dst', type=str, required=True, help='destination repo')    
    parser.add_argument('-crypto', type=str, required=True, help='crypto')    
    return parser.parse_args()

if __name__ == '__main__':
    try:
        args = ArgParsing()
        df = pd.read_csv(args.file)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        years = df['DATETIME'].dt.year.unique()

        for year in years:
            df_year = df[df['DATETIME'].dt.year == year].copy()
            df_year.sort_values(by='DATETIME', inplace=True)
            df_year.to_csv(f'{args.dst}/{args.crypto}_{str(year)}.csv', index=False)

    except Exception as error:
        print(error)