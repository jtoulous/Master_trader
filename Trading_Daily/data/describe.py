import pandas as pd

if __name__ == '__main__':
    try:
        df = pd.read_csv('data/EURUSD/EURUSD_preprocessed.csv')
        label_counts = df['LABEL'].value_counts()
        print(label_counts)

    except Exception as error:
        print (error)