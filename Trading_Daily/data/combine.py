import pandas as pd
import argparse as ap

def ArgParsing():
    parser = ap.ArgumentParser(
        prog='concat datafiles',
        description='combine multiple datafiles into one'
    )
    parser.add_argument('datafiles', type=str, nargs='+', help='raw datafiles to combine')
    parser.add_argument('-dst', type=str, default='output.csv', help='destination file')
    return parser.parse_args()

if __name__ == '__main__':
    try:
        args = ArgParsing()
        dataframes = []
        for datafile in args.datafiles:
            dataframes.append(pd.read_csv(datafile))
        combined_df = pd.concat(dataframes)
        combined_df = combined_df.sort_values(by='DATETIME')
        combined_df.to_csv(args.dst, index=False)

    except Exception as error:
        print(error)