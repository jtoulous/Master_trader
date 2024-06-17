## Objectives
Predict whether positions taken on a buy will be winning or losing positions.

## Algorithm Explanation
1. Preprocessing:
   1. We have a data file, `srcs/data/EURUSD/EURUSD_DATA.csv`, which contains every value of the EURUSD for each minute over the past three years (OPEN, HIGH, LOW, CLOSE). Using these values, I will calculate all the indicators that will be used as features.
   2. I will determine the label for each line, which will be either 'WIN' or 'LOSS', by calculating the stop-loss and take-profit using the ATR (Average True Range). According to the -lifespan argument, I will observe the subsequent values and check whether the stop-loss or the take-profit is hit first.
   3. Finally, I will scale my features and save the preprocessed data in `srcs/data/EURUSD/EURUSD_preprocessed.csv` to save time on future runs.

2. Training:
   1. Using scikit-learn, I train 5 different models (MLP, XGB, Random forest, Logistic regression, Gradient boosting), with my dataframe EURUSD_preprocessed.csv. Under sampling the losses is necessary since we have way more losses then wins in the dataframe.
   2. I save the trained models with joblib
   
3. Prediction:
   1. I preprocess the given data file if needed.
   2. I load my 5 trained models with joblib.
   3. I make the predictions for each model.
   4. I Combine the predictions of the 5 models (will be a 'WIN' only if the 5 models have predicted a 'WIN')
   5. I print the result.
