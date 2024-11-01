import pandas as pd
from datetime import datetime

from sklearn.preprocessing import StandardScaler, RobustScaler
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
    today_date = pd.to_datetime(datetime.today().strftime('%d/%m/%Y'), format='%d/%m/%Y')
    prev_date = pd.to_datetime(date, format='%d/%m/%Y') - pd.Timedelta(days=1)
    df = dataframe.copy()
    df = date_to_features(df)
    df = ATR(df, GetArg('atr'))
    df = SMA(df, GetArg('sma'))
    df = EMA(df, GetArg('ema'))
#    df = RSI(df, GetArg('rsi'))
    df = Bollinger_bands(df, GetArg('blg'))
    df = MACD(df, GetArg('macd'))
#    df = STO(df, GetArg('sto'))
    df = ROC(df)
#    df = DMI(df, GetArg('dmi'))
    df['GROWTH'] = (df['CLOSE'] - df['OPEN']) / df['OPEN'] * 100
    rectify_df = df.copy()

    df['LABEL'] = df[pred_type].shift(-1)
    df = df.ffill()
    df = df[(df['DATETIME'] != date) & (df['DATETIME'] != today_date)]
    features = list(df.columns)
    features.remove('DATETIME')
    features_df = df[features]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features_df)
    df[features] = pd.DataFrame(scaled_values, index=df.index, columns=features)
    label_mean = scaler.mean_[features.index('LABEL')]
    label_scale = scaler.scale_[features.index('LABEL')]

    features.remove('LABEL')
    X_predict = df[features][df['DATETIME'] == prev_date]
    df = df[df['DATETIME'] != prev_date]
    X_train = df.iloc[:-2][features]
    y_train = df.iloc[:-2]['LABEL']
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

    prediction = (mlp_pred_denorm[0] + gbc_pred_denorm[0] + rfc_pred_denorm[0]) / 3
#    print(f' {pred_type} uncorrected = {prediction}')
#    prediction = RectifyEstimation(pred_type, rectify_df, prev_date, MLP, GBC, RFC, label_mean, label_scale, prediction)

    return prediction


def RectifyEstimation(pred_type, dataframe, date, MLP, GBC, RFC, label_mean, label_scale, prediction):
    df_1 = dataframe[:-2].copy()
    df_2 = dataframe[:-2].copy()

    breakpoint()
    features = list(df_1.columns)
    features.remove('DATETIME')
    features_df = df_1[features]
    scaler_df1 = StandardScaler()
    tmp_df = pd.DataFrame(scaler_df1.fit_transform(features_df), columns=features)
    df_1[features] = tmp_df

    for idx, row in df_1.iterrows():
        if idx == len(df_1) - 1:
            break
        row_date = row['DATETIME']
        df_predict = row[features].to_frame().T
        mlp_pred = MLP.predict(df_predict)[0] * label_scale + label_mean
        rfc_pred = RFC.predict(df_predict)[0] * label_scale + label_mean
        gbc_pred = GBC.predict(df_predict)[0] * label_scale + label_mean

        pred = (mlp_pred + gbc_pred + rfc_pred) / 3
        actual_value = float(df_2[df_2['DATETIME'] == row_date + pd.Timedelta(days=1)][pred_type].iloc[0])
        df_2.loc[df_2['DATETIME'] == row_date, 'LABEL'] = pred - actual_value
    
    features = list(df_2.columns)
    features.remove('DATETIME')
    features_df = df_2[features]
    scaler_df2 = RobustScaler()
    tmp_df = pd.DataFrame(scaler_df2.fit_transform(features_df), columns=features)
    df_2[features] = tmp_df
    label_mean_df2 = scaler_df2.center_[features.index('LABEL')]
    label_scale_df2 = scaler_df2.scale_[features.index('LABEL')]
    features.remove('LABEL')

    X_predict = df_2[df_2['DATETIME'] == date][features]
    X_train = df_2[df_2['DATETIME'] != date][features][:-1]
    y_train = df_2[df_2['DATETIME'] != date]['LABEL'][:-1]

    RFC = RandomForestRegressor(n_estimators=100, random_state=42)
    GBC = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    MLP = MLPRegressor(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
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

    mlp_pred_denorm = mlp_pred * label_scale_df2 + label_mean_df2
    gbc_pred_denorm = gbc_pred * label_scale_df2 + label_mean_df2
    rfc_pred_denorm = rfc_pred * label_scale_df2 + label_mean_df2

    correction = (mlp_pred_denorm[0] + gbc_pred_denorm[0] + rfc_pred_denorm[0]) / 3
    return prediction + correction