import pandas as pd
import sys
import argparse as ap
from utils.indicators import calc_indicators
from utils.preprocessing import calc_labels

def ArgParsing():
    parser = ap.ArgumentParser(
        prog='trading algo',
        description='predictive model for trading'
    )
    parser.add_argument('datafile', type=str, help='datafile to describe')

    parser.add_argument('-lifespan', type=int, default=10, help='lifespan of the trade in days')
    parser.add_argument('-risk', type=float, default=1, help='percentage of capital for the stop-loss')
    parser.add_argument('-profit', type=float, default=3, help='percentage of capital for the take-profit')
    parser.add_argument('-atr', type=int, default=14, help='periods used for calculating ATR')
    parser.add_argument('-ema', type=int, default=50, help='periods used for calculating EMA')
    parser.add_argument('-rsi', type=int, default=14, help='periods used for calculating RSI')
    parser.add_argument('-sto', type=int, default=[14, 3], nargs=2, help='periods used for calculating STO')
    parser.add_argument('-sma', type=int, default=50, help='periods used for calculating SMA')
    parser.add_argument('-wma', type=int, default=20, help='periods used for calculating WMA')
    parser.add_argument('-dmi', type=int, default=14, help='periods used for calculating DMI')
    parser.add_argument('-blg', type=int, default=[20, 2], nargs=2, help='periods and nb_stddev used for calculating Bollingers bands')
    parser.add_argument('-macd', type=int, default=[12, 26, 9], nargs=3, help='periods(short, long, signal) used for calculating MACD')
    parser.add_argument('-cci', type=int, default=20, help='periods used for calculating CCI')
    parser.add_argument('-ppo', type=int, default=[12, 26, 9], nargs=3, help='periods(short, long, signal) used for calculating PPO')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = ArgParsing()
        df = pd.read_csv(sys.argv[1])
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df = df.sort_values(by='DATETIME')
        df  = calc_indicators(df, args) 
        df = calc_labels(df, args)
        label_counts = df['LABEL'].value_counts()
        print(label_counts)

    except Exception as error:
        print (error)