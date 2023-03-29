import pickle

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(transactions):
    return transactions[['Time','Amount','V2', 'V3', 'V4', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V26', 'V27']]
    


def predict(transactions):
    # load model
    with open('nb_model_v2', 'rb') as f:
        nb = pickle.load(f)

    y_pred= nb.predict(preprocess(transactions))

    return y_pred