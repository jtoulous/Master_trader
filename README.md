## Objectives
Predict if positions taken on a buy will be winning positions or losing positions 

## Algo explanation
1. Preprocessing:<br>
   1. we have a datafile `srcs/data/EURUSD/EURUSD_DATA.csv`, wich as every value of the EURUSD for every minute in the last 3 years (OPEN, HIGH, LOW, CLOSE),
with these values I calculate every indicators that will be used as my features.
   2. I determine the label for each line wich will either be a 'WIN' or 'LOSS' by calculating my stop-loss
and take-profit using the ATR, then according to the argument `-lifespan` i will observe the next values and check if it either hits the
stop-loss or the take-profit first.
   3. Finally, I will scale my features and save the preprocessed data in `srcs/data/EURUSD/EURUSD_preprocessed.cv`, to save time on the next runs
