## Objectives
Predict whether positions taken on a buy will be winning or losing positions.

## Algorithm Explanation
1. Preprocessing:
   1. We have a data file, `srcs/data/EURUSD/EURUSD_DATA.csv`, which contains every value of the EURUSD for each minute over the past three years (OPEN, HIGH, LOW, CLOSE). Using these values, I will calculate all the indicators that will be used as features.
   2. I will determine the label for each line, which will be either 'WIN' or 'LOSS', by calculating the stop-loss and take-profit using the ATR (Average True Range). According to the -lifespan argument, I will observe the subsequent values and check whether the stop-loss or the take-profit is hit first.
   3. Finally, I will scale my features and save the preprocessed data in `srcs/data/EURUSD/EURUSD_preprocessed.csv` to save time on future runs.
