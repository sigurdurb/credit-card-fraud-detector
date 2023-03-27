import pickle

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(transactions):
    transactions_procesed = transactions.copy()
    transactions_procesed["Time-Mod"]=(transactions_procesed["Time"]%(60*60*24))/(60*60*24)
    transactions_procesed.pop('Time')

    return transactions_procesed
 

def predict(transactions):
    # load model
    with open('random_forest_detector/rf_detector', 'rb') as f:
        rfc = pickle.load(f)

    y_pred= rfc.predict(preprocess(transactions))

    return y_pred
