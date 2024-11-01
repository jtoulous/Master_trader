import pandas as pd
import subprocess
from utils.arguments import ActiveCryptos

if __name__ == '__main__':
    try:
        for crypto in ActiveCryptos():
            command_1 = [
                'python',
                'combine.py',
                f'CRYPTOS/{crypto}/raw/{crypto}_2020.csv',
                f'CRYPTOS/{crypto}/raw/{crypto}_2021.csv',
                f'CRYPTOS/{crypto}/raw/{crypto}_2022.csv',
                f'CRYPTOS/{crypto}/raw/{crypto}_2023.csv',
                '-dst',
                f'CRYPTOS/{crypto}/test_train.csv',
            ]

            command_2 = [
                'python',
                'combine.py',
                f'CRYPTOS/{crypto}/raw/{crypto}_2024.csv',
                '-dst',
                f'CRYPTOS/{crypto}/test_predict.csv'
            ]
            subprocess.run(command_1)
            subprocess.run(command_2)

    except Exception as error:
        print(error)