## Objectives
Predict whether positions taken on a buy will be winning or losing positions.

## Usage
1. Run `make` to create a virtual environment and install dependencies.
2. Run `python train.py -EURUSD data/EURUSD/EURUSD_data.csv` to start the training.
3. Run `python predict_test.py -EURUSD data/EURUSD/EURUSD_preprocessed.csv` to run prediction tests on `EURUSD_preprocessed.csv`. The script `predict_test.py` will make predictions for every element in the data file, compare them to the true labels, and display a summary of the prediction accuracy.
4. Run `python predict.py -EURUSD data/EURUSD/EURUSD_preprocessed.csv` to run predictions on `EURUSD_preprocessed.csv`. The script `predict.py` will make predictions for every element in the data file without comparing them to the true labels, allowing you to use a CSV file that isn't preprocessed but contains "DATETIME,OPEN,HIGH,LOW,CLOSE".

## Algorithm Explanation

1. **Preprocessing:**
   1. We have a data file, `srcs/data/EURUSD/EURUSD_DATA.csv`, which contains every value of the EURUSD for each minute over the past three years (OPEN, HIGH, LOW, CLOSE). Using these values, I will calculate all the indicators that will be used as features.
   2. I will determine the label for each line, which will be either 'WIN' or 'LOSS', by calculating the stop-loss and take-profit using the ATR (Average True Range). According to the `-lifespan` argument, I will observe the subsequent values and check whether the stop-loss or the take-profit is hit first.
   3. Finally, I will scale my features and save the preprocessed data in `srcs/data/EURUSD/EURUSD_preprocessed.csv` to save time on future runs.

2. **Training:**
   1. Using scikit-learn, I train 5 different models (MLP, XGB, Random Forest, Logistic Regression, Gradient Boosting) with the `EURUSD_preprocessed.csv` dataframe. Under-sampling the losses is necessary since the dataframe contains significantly more losses than wins.
   2. I save the trained models using joblib.
   
3. **Prediction:**
   1. I preprocess the given data file if needed.
   2. I load my 5 trained models using joblib.
   3. I make predictions with each model.
   4. I combine the predictions of the 5 models (a result will be 'WIN' only if all 5 models predict 'WIN').
   5. I print the result.
