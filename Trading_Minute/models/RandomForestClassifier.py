import sys
import pandas as pd
import joblib

from colorama import Fore, Style
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


if __name__ == '__main__':
    try:
        print(f'{Fore.GREEN} ===> Reading data...')
        dataframe = pd.read_csv(sys.argv[1], index_col=False)
        features = list(dataframe.columns)
        features.remove('LABEL')
        features.remove('DATETIME')
        y = dataframe['LABEL']
        X = dataframe[features]
        print(' ===> Done')

        print(' ===> Splitting data...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(' ===> Done')

        print(' ===> Under sampling...')
        under_sampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)
        print(' ===> Done')

        print(' ===> Training forest classifier...')
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_resampled, y_train_resampled)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/random_forest_model_{sys.argv[2]}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')

    except Exception as error:
        print(f'{Fore.RED}{error}{Style.RESET_ALL}')