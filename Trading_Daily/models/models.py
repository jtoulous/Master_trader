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


warnings.filterwarnings("ignore", category=ConvergenceWarning)

def GradientBoosting(df, currency_pair):
        print(f'{Fore.GREEN} ===> Reading data...')
        X, y = GetXnY(df.copy())
        print(' ===> Done')

#        print(f' ===> Splitting data...')
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#        print(' ===> Done')

        print(' ===> Over sampling...')
        over_sampler = RandomOverSampler(sampling_strategy='auto')
        X_resampled, y_resampled = over_sampler.fit_resample(X, y)
        print(' ===> Done')

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
 
        print(' ===> Cross validation...')
        scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
        print(f'   ==> Cross-Validation Scores: {scores}')
        print(f'   ==> Average Accuracy: {scores.mean()}')
        print(' ===> Done')

        print(' ===> Training...')
        model.fit(X_resampled, y_resampled)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/architectures/gradient_boosting_{currency_pair}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')


def LogReg(df, currency_pair):
        print(f'{Fore.GREEN} ===> Reading data...')
        X, y = GetXnY(df.copy())
        print(' ===> Done')
        
#        print(' ===> Splitting data...')
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#        print(' ===> Done')

        print(' ===> Under sampling...')
        over_sampler = RandomOverSampler(sampling_strategy='auto')
        X_resampled, y_resampled = over_sampler.fit_resample(X, y)
        print(' ===> Done')

        model = LogisticRegression(max_iter=1000)
        
        print(' ===> Cross validation...')
        scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
        print(f'   ==> Cross-Validation Scores: {scores}')
        print(f'   ==> Average Accuracy: {scores.mean()}')
        print(' ===> Done')
        
        print(' ===> Training logistic regression...')
        model.fit(X_resampled, y_resampled)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/architectures/logreg_{currency_pair}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')


def MLP(df, currency_pair):
        print(f'{Fore.GREEN} ===> Reading data...')
        X, y = GetXnY(df.copy())
        print(' ===> Done')
        
#        print(f' ===> Splitting data...')
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#        print(' ===> Done')

        print(' ===> Over sampling...')
        over_sampler = RandomOverSampler(sampling_strategy='auto')
        X_resampled, y_resampled = over_sampler.fit_resample(X, y)
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
        
        print(' ===> Cross validation...')
        scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
        print(f'   ==> Cross-Validation Scores: {scores}')
        print(f'   ==> Average Accuracy: {scores.mean()}')
        print(' ===> Done')
        
        print(' ===> Training MLP...')
        model.fit(X_resampled, y_resampled)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/architectures/mlp_{currency_pair}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')


def RFClassifier(df, currency_pair):
        print(f'{Fore.GREEN} ===> Reading data...')
        X, y = GetXnY(df.copy())
        print(' ===> Done')

#        print(' ===> Splitting data...')
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#        print(' ===> Done')

        print(' ===> Over sampling...')
        over_sampler = RandomOverSampler(sampling_strategy='auto')
        X_resampled, y_resampled = over_sampler.fit_resample(X, y)
        print(' ===> Done')

        model = RandomForestClassifier(n_estimators=100, random_state=42)

        print(' ===> Cross validation...')
        scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
        print(f'   ==> Cross-Validation Scores: {scores}')
        print(f'   ==> Average Accuracy: {scores.mean()}')
        print(' ===> Done')

        print(' ===> Training forest classifier...')
        model.fit(X_resampled, y_resampled)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/architectures/random_forest_model_{currency_pair}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')


def XGB(df, currency_pair):
        print(f'{Fore.GREEN} ===> Reading data...')
        X, y = GetXnY(df.copy())
        print(' ===> Done')
        
        print(' ===> Encoding labels...')
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y) 
        print(' ===> Done')

#        print(' ===> Splitting data...')
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#        print(' ===> Done')

        print(' ===> Over sampling...')
        over_sampler = RandomOverSampler(sampling_strategy='auto')
        X_resampled, y_resampled = over_sampler.fit_resample(X, y)
        print(' ===> Done')

        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        
        print(' ===> Cross validation...')
        scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
        print(f'   ==> Cross-Validation Scores: {scores}')
        print(f'   ==> Average Accuracy: {scores.mean()}')
        print(' ===> Done')

        print(' ===> Training xgb...')
        model.fit(X_resampled, y_resampled)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/architectures/xgb_{currency_pair}.pkl')
        joblib.dump(label_encoder, f'models/architectures/xgb_label_encoder_{currency_pair}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')


def GetXnY(dataframe):
        features = list(dataframe.columns)
        features.remove('LABEL')
        features.remove('DATETIME')
        features.remove('HIGH')
        features.remove('LOW')
        features.remove('CLOSE')
        features.remove('VOLUME')

        return dataframe[features], dataframe['LABEL']