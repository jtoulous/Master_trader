import sys
import joblib
import pandas as pd
import warnings
from colorama import Fore, Style

import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

from utils.arguments import GetFeatures

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def GradientBoosting(df, currency_pair, crossval, data_type):
        print(f'{Fore.GREEN} ===> Reading data...')
        X, y = df[GetFeatures()], df['LABEL']
        print(' ===> Done')

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
 
        if crossval == True:
                print(' ===> Cross validation...')
                scores = cross_val_score(model, X, y, cv=5)
                print(f'   ==> Cross-Validation Scores: {scores}')
                print(f'   ==> Average Accuracy: {scores.mean()}')
                print(' ===> Done')

        print(' ===> Training...')
        model.fit(X, y)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/architectures/gradient_boosting_{currency_pair}_{data_type}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')


def LogReg(df, currency_pair, crossval, data_type):
        print(f'{Fore.GREEN} ===> Reading data...')
        X, y = df[GetFeatures()], df['LABEL']
        print(' ===> Done')

        model = LogisticRegression(max_iter=1000)
        
        if crossval == True:
                print(' ===> Cross validation...')
                scores = cross_val_score(model, X, y, cv=5)
                print(f'   ==> Cross-Validation Scores: {scores}')
                print(f'   ==> Average Accuracy: {scores.mean()}')
                print(' ===> Done')
        
        print(' ===> Training logistic regression...')
        model.fit(X, y)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/architectures/logreg_{currency_pair}_{data_type}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')


def MLP(df, currency_pair, crossval, data_type):
        print(f'{Fore.GREEN} ===> Reading data...')
        X, y = df[GetFeatures()], df['LABEL']
        print(' ===> Done')

        model = MLPClassifier(
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
        
        if crossval == True:
                print(' ===> Cross validation...')
                scores = cross_val_score(model, X, y, cv=5)
                print(f'   ==> Cross-Validation Scores: {scores}')
                print(f'   ==> Average Accuracy: {scores.mean()}')
                print(' ===> Done')
        
        print(' ===> Training MLP...')
        model.fit(X, y)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/architectures/mlp_{currency_pair}_{data_type}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')


def RFClassifier(df, currency_pair, crossval, data_type):
        print(f'{Fore.GREEN} ===> Reading data...')
        X, y = df[GetFeatures()], df['LABEL']
        print(' ===> Done')

        model = RandomForestClassifier(n_estimators=100, random_state=42)

        if crossval == True:
                print(' ===> Cross validation...')
                scores = cross_val_score(model, X, y, cv=5)
                print(f'   ==> Cross-Validation Scores: {scores}')
                print(f'   ==> Average Accuracy: {scores.mean()}')
                print(' ===> Done')

        print(' ===> Training forest classifier...')
        model.fit(X, y)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/architectures/random_forest_model_{currency_pair}_{data_type}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')


def XGB(df, currency_pair, crossval, data_type):
        print(f'{Fore.GREEN} ===> Reading data...')
        X, y = df[GetFeatures()], df['LABEL']
        print(' ===> Done')
        
        print(' ===> Encoding labels...')
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y) 
        print(' ===> Done')

        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

        if crossval == True:     
                print(' ===> Cross validation...')
                scores = cross_val_score(model, X, y, cv=5)
                print(f'   ==> Cross-Validation Scores: {scores}')
                print(f'   ==> Average Accuracy: {scores.mean()}')
                print(' ===> Done')

        print(' ===> Training xgb...')
        model.fit(X, y)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/architectures/xgb_{currency_pair}.pkl')
        joblib.dump(label_encoder, f'models/architectures/xgb_label_encoder_{currency_pair}_{data_type}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')