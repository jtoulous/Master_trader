## Objectives
Predict whether positions taken on a buy will be winning or losing positions.

## Usage
1. Run `make` to create a virtual environment and install dependencies.

2. Run `python train.py` to start the training wich will use all your db for training, or run `python train.py -test` to train the models on historical data from 2020-2023 in order to use the predict_test.py for validation, that will run predictions on 2024.

3. Run `python predict.py -estimation` to run predictions for today, which is less reliable than running predictions on yesterday since the estimation option will use some other predictions models to predict the HIGH, LOW and CLOSE of the current day in order to calculate the indicators that will be used for the final prediction.

4. Run `python predict.py -date 13/03/2021`, to run prediction on a historical date without using the estimation system for the HIGH, LOW and CLOSE, it will simply use their true value stored in the historical db, or you can add `-estimation` if you wish to see the less reliable result using the estimation system.

5. Run `python predict_test.py` to run prediction tests, wich will run predictions on all the data from 2024. This script is for validating the training by making predictions on unseen data, so if you use this script you need to train the models using `python train.py -test` wich will exclude the data from 2024 from the training.
