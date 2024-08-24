import pandas as pd
import numpy as np

from scipy.signal import hilbert
from .tools import printLog, printError, printInfo

def date_to_features(dataframe):
    printLog('Converting dates...')    
    dataframe['DAY'] = dataframe['DATETIME'].dt.dayofweek + 1
    dataframe['HOUR'] = dataframe['DATETIME'].dt.hour
    dataframe['MINUTE'] = dataframe['DATETIME'].dt.minute
    #potentiellement rajouter le mois ou ballec
    printLog('Done')
    return dataframe

def ATR(dataframe, periods):
    printLog('Calculating ATR...')
    tr = []
    prev_row = None

    for idx, row in dataframe.iterrows():
        if idx == 0:
            tr_val = row['HIGH'] - row['LOW']
        else:
            tr_val = max(row['HIGH'] - row['LOW'], abs(row['HIGH'] - prev_row['CLOSE']), abs(row['LOW'] - prev_row['CLOSE']))
        tr.append(tr_val)
        prev_row = row

    tr = pd.Series(tr)
    dataframe['ATR'] = tr.rolling(window=periods, min_periods=1).mean()
    printLog(f'Done')
    return dataframe

def EMA(dataframe, periods):
    printLog(f'Calculating EMA...')
    dataframe['EMA'] = dataframe['CLOSE'].rolling(window=periods, min_periods=1).mean()
    printLog(f'Done')
    return dataframe

def RSI(dataframe, periods):
    printLog(f'Calculating RSI...')
    delta = dataframe['CLOSE'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (delta.where(delta < 0, 0)).fillna(0)
    loss = -loss

    avg_gain = gain.rolling(window=periods, min_periods=1).mean()
    avg_loss = loss.rolling(window=periods, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    dataframe['RSI'] = rsi.bfill()
    printLog(f'Done')
    return dataframe

def STO(dataframe, periods):
    printLog(f'Calculating STO...')
    k_periods = periods[0]
    d_periods = periods[1]

    low_min = dataframe['LOW'].rolling(window=k_periods, min_periods=1).min()
    high_max = dataframe['HIGH'].rolling(window=k_periods,min_periods=1).max()
    K = ((dataframe['CLOSE'] - low_min) / (high_max - low_min)) * 100
    D = K.rolling(window=d_periods, min_periods=1).mean()

    dataframe['K(sto)'] = K.bfill()
    dataframe['D(sto)'] = D.bfill()
    printLog('Done')
    return dataframe

def SMA(dataframe, periods):
    printLog(f'Calculating SMA...')
    dataframe['SMA'] = dataframe['CLOSE'].rolling(window=periods, min_periods=1).mean()
    dataframe['SMA'] = dataframe['SMA'].bfill()
    printLog('Done')
    return dataframe

def WMA(dataframe, periods):
    printLog(f'Calculating WMA...')
    weights = pd.Series(np.arange(1, periods + 1) / np.sum(np.arange(1, periods + 1)))
    weighted_close = dataframe['CLOSE'].rolling(window=periods).apply(lambda x: (x * weights).sum(), raw=True)
    dataframe['WMA'] = weighted_close.bfill()
    printLog('Done')
    return dataframe

def DMI(dataframe, periods):
    printLog(f'Calculating DMI...')
    prev_row = None
    dm_plus = []
    dm_minus = []

    for idx, row in dataframe.iterrows():
        if prev_row is not None:
            dm_plus_val = row['HIGH'] - prev_row['HIGH'] \
                            if row['HIGH'] - prev_row['HIGH'] > prev_row['LOW'] - row['LOW'] \
                            and row['HIGH'] - prev_row['HIGH'] > 0 \
                            else 0
            dm_minus_val = prev_row['LOW'] - row['LOW'] \
                            if prev_row['LOW'] - row['LOW'] > row['HIGH'] - prev_row['HIGH'] \
                            and prev_row['LOW'] - row['LOW'] > 0 \
                            else 0
        else:
            dm_plus_val = 0
            dm_minus_val = 0
        dm_plus.append(dm_plus_val)
        dm_minus.append(dm_minus_val)
        prev_row = row

    dm_plus = pd.Series(dm_plus).bfill()
    dm_minus = pd.Series(dm_minus).bfill()
    dm_diff = abs(dm_plus - dm_minus)
    dm_sum = dm_plus + dm_minus
    dx = dm_diff / dm_sum
    adx = dx.rolling(window=periods, min_periods=1).mean().bfill()
    
    dataframe['DM+'] = dm_plus
    dataframe['DM-'] = dm_minus
    dataframe['ADX'] = adx
    printLog(f'Done')
    return dataframe

def Bollinger_bands(dataframe, args):
    printLog(f'Calculating Bollingers bands...')
    periods = args[0]
    num_std_dev = args[1]

    TMP_SMA = dataframe['CLOSE'].rolling(window=periods, min_periods=1).mean()
    TMP_STD = dataframe['CLOSE'].rolling(window=periods, min_periods=1).std()
    u_band = TMP_SMA + (TMP_STD * num_std_dev)
    l_band = TMP_SMA - (TMP_STD * num_std_dev)

    dataframe['U-BAND'] = u_band.bfill()
    dataframe['L-BAND'] = l_band.bfill()
    printLog('Done')
    return dataframe

def MACD(dataframe, args):
    printLog(f'Calculating MACD...')
    short_period = args[0]
    long_period = args[1]
    signal_period = args[2]

    ema_short = dataframe['CLOSE'].rolling(window=short_period, min_periods=1).mean()
    ema_long = dataframe['CLOSE'].rolling(window=long_period, min_periods=1).mean()
    macd_line = ema_short - ema_long
    macd_signal = macd_line.ewm(span=signal_period, min_periods=1).mean()
    macd_histo = macd_line - macd_signal

    dataframe['MACD_LINE'] = macd_line.bfill()
    dataframe['MACD_SIGNAL'] = macd_signal.bfill()
    dataframe['MACD_HISTO'] = macd_histo.bfill()
    printLog('Done')
    return dataframe

def Hilberts_transform(dataframe):
    printLog(f'Calculating Hilberts transform...')
    analytic_signal = hilbert(dataframe['CLOSE'])
    dataframe['Hilbert_Transform'] = analytic_signal.imag
    dataframe['Hilbert_Transform'] = dataframe['Hilbert_Transform'].bfill()
    printLog('Done')
    return dataframe

def CCI(dataframe, periods):
    printLog('Calculating CCI...')
    TP = (dataframe['HIGH'] + dataframe['LOW'] + dataframe['CLOSE']) / 3
    SMA_TP = TP.rolling(window=periods, min_periods=1).mean()
    MD = TP.rolling(window=periods, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))))

    CCI = (TP -SMA_TP) / (0.015 * MD)
    dataframe['CCI'] = CCI.bfill()
    printLog('Done')
    return dataframe

def PPO(dataframe, periods):
    printLog('Calculating PPO...')
    short_period = periods[0]
    long_period = periods[1]
    signal_period = periods[2]

    short_ema = dataframe['CLOSE'].ewm(span=short_period, min_periods=1).mean()
    long_ema = dataframe['CLOSE'].ewm(span=long_period, min_periods=1).mean()
    ppo_line = ((short_ema - long_ema) / long_ema) * 100
    signal_line = ppo_line.ewm(span=signal_period, min_periods=1).mean()
    
    dataframe['PPO_LINE'] = ppo_line.bfill()
    dataframe['PPO_SIGNAL'] = signal_line.bfill()
    printLog('Done')
    return dataframe

def ROC(dataframe):
    printLog('Calculating ROC...')
    dataframe['ROC'] = dataframe['CLOSE'].pct_change()
    dataframe['ROC'] = dataframe['ROC'].bfill()
    printLog('Done')
    return dataframe

def feature_engineering(dataframe):
    printLog('Feature engineering...')
    dataframe['ATR_Lagged'] = dataframe['ATR'].shift(1).bfill()
    dataframe['EMA_Lagged'] = dataframe['EMA'].shift(1).bfill()
    dataframe['RSI_Lagged'] = dataframe['RSI'].shift(1).bfill()
    dataframe['Momentum'] = dataframe['CLOSE'] - dataframe['CLOSE'].shift(1).bfill()
    dataframe['MACD_Difference'] = dataframe['MACD_LINE'] - dataframe['MACD_SIGNAL'].bfill()
    dataframe['Bollinger_Width'] = dataframe['U-BAND'] - dataframe['L-BAND'].bfill()
    dataframe['EMA_SMA_Ratio'] = dataframe['EMA'] / dataframe['SMA'].bfill()
    printLog('Done')
    return dataframe

def calc_indicators(dataframe, args):
    dataframe = date_to_features(dataframe)
    dataframe = ATR(dataframe, args.atr)
    dataframe = EMA(dataframe, args.ema)
    dataframe = RSI(dataframe, args.rsi)
    dataframe = STO(dataframe, args.sto)
    dataframe = SMA(dataframe, args.sma)
    dataframe = WMA(dataframe, args.wma)
    dataframe = DMI(dataframe, args.dmi)
    dataframe = Bollinger_bands(dataframe, args.blg)
    dataframe = MACD(dataframe, args.macd)
    dataframe = Hilberts_transform(dataframe)
    dataframe = CCI(dataframe, args.cci)
    dataframe = PPO(dataframe, args.ppo)
    dataframe = ROC(dataframe)
    dataframe = feature_engineering(dataframe)
    return dataframe