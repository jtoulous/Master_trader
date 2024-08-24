import sys
import joblib
import pandas as pd

from colorama import Fore, Style
from sklearn.ensemble import GradientBoostingClassifier
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

        print(f' ===> Splitting data...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(' ===> Done')

        print(' ===> Over sampling...')
        over_sampler = RandomOverSampler(sampling_strategy='auto')
        X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)
        print(' ===> Done')

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
 
        print(' ===> Cross validation...')
        scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5)
        print(f'   ==> Cross-Validation Scores: {scores}')
        print(f'   ==> Average Accuracy: {scores.mean()}')
        print(' ===> Done')

        print(' ===> Training...')
        model.fit(X_train_resampled, y_train_resampled)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/gradient_boosting_{sys.argv[2]}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')

    except Exception as error:
        print(f'{Fore.RED}{error}{Style.RESET_ALL}')