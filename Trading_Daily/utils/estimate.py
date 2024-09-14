import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from .indicators import ATR, SMA, EMA, RSI, MACD, DMI
from .indicators import Bollinger_bands, STO, ROC, date_to_features
from .arguments import GetArg

def Cross_Val(model, X, y, cv, pred_type):
    print(f' ===> {pred_type} Cross validation...')
    scores = cross_val_score(model, X, y, cv=cv)
    print(f'   ==> Cross-Validation Scores: {scores}')
    print(f'   ==> Average Accuracy: {scores.mean()}')
    print(' ===> Done')


def Estimate(dataframe, date, pred_type):
    date = pd.to_datetime(date, format='%d/%m/%Y')
    df = dataframe.copy()
    df = date_to_features(df)
    df = ATR(df, GetArg('atr'))
    df = SMA(df, GetArg('sma'))
    df = EMA(df, GetArg('ema'))
    df = RSI(df, GetArg('rsi'))
    df = Bollinger_bands(df, GetArg('blg'))
    df = MACD(df, GetArg('macd'))
    df = STO(df, GetArg('sto'))
    df = ROC(df)
    df = DMI(df, GetArg('dmi'))
    df['GROWTH'] = (df['CLOSE'] - df['OPEN']) / df['OPEN'] * 100
    df['LABEL'] = df[pred_type].shift(-1)
   
    features = list(df.columns)
    features.remove('DATETIME')
    features_df = df[features]
    scaler = StandardScaler()
    tmp_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features)
    df[features] = tmp_df
    label_mean = scaler.mean_[features.index('LABEL')]
    label_scale = scaler.scale_[features.index('LABEL')]

    features.remove('LABEL')
    X_predict = df[features][df['DATETIME'] == date]
    df = df[df['DATETIME'] != date]
    X_train = df.iloc[:-1][features]
    y_train = df.iloc[:-1]['LABEL']

    RFC = RandomForestRegressor(n_estimators=100, random_state=42)
    GBC = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    MLP = MLPRegressor(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=300,
        shuffle=True,
        random_state=42,
        verbose=False
    )

#    Cross_Val(MLP, X_train, y_train, 5, 'MLP')
#    Cross_Val(RFC, X_train, y_train, 5, 'RFC')
#    Cross_Val(GBC, X_train, y_train, 5, 'GBC')

    MLP.fit(X_train, y_train)
    GBC.fit(X_train, y_train)
    RFC.fit(X_train, y_train)

    mlp_pred = MLP.predict(X_predict)
    gbc_pred = GBC.predict(X_predict)
    rfc_pred = RFC.predict(X_predict)

    mlp_pred_denorm = mlp_pred * label_scale + label_mean
    gbc_pred_denorm = gbc_pred * label_scale + label_mean
    rfc_pred_denorm = rfc_pred * label_scale + label_mean

    return (mlp_pred_denorm[0] + gbc_pred_denorm[0] + rfc_pred_denorm[0]) / 3