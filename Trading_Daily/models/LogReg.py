import sys
import pandas as pd
import joblib

from colorama import Fore, Style
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import RandomOverSampler


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
        over_sampler = RandomOverSampler(sampling_strategy='auto')
        X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)
        print(' ===> Done')

        model = LogisticRegression(max_iter=1000)
        
        print(' ===> Cross validation...')
        scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5)
        print(f'   ==> Cross-Validation Scores: {scores}')
        print(f'   ==> Average Accuracy: {scores.mean()}')
        print(' ===> Done')
        
        print(' ===> Training logistic regression...')
        model.fit(X_train_resampled, y_train_resampled)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/logreg_{sys.argv[2]}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')

    except Exception as error:
        print(f'{Fore.RED}{error}{Style.RESET_ALL}')