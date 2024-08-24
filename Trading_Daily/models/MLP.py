import sys
import pandas as pd
import joblib
import warnings

from colorama import Fore, Style
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import RandomOverSampler


warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
        scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5)
        print(f'   ==> Cross-Validation Scores: {scores}')
        print(f'   ==> Average Accuracy: {scores.mean()}')
        print(' ===> Done')
        
        print(' ===> Training MLP...')
        model.fit(X_train_resampled, y_train_resampled)
        print(' ===> Done')
      
        print(' ===> Saving model...')
        joblib.dump(model, f'models/mlp_{sys.argv[2]}.pkl')
        print(f' ===> Done{Style.RESET_ALL}')

    except Exception as error:
        print(f'{Fore.RED}{error}{Style.RESET_ALL}')