import pickle

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(transactions):
    transactions_procesed = transactions.copy()
    transactions_procesed.pop('Time')
    transactions_procesed.pop('V21')
    transactions_procesed.pop('V9')
    transactions_procesed.pop('V8')
    transactions_procesed.pop('V20')
    transactions_procesed.pop('V1')
    transactions_procesed.pop('V18')
    transactions_procesed.pop('V15')
    transactions_procesed.pop('V28')
    transactions_procesed.pop('V23')
    transactions_procesed.pop('Amount')
    transactions_procesed.pop('V25')
    transactions_procesed.pop('V26')
    transactions_procesed.pop('V6')
    transactions_procesed.pop('V22')
    transactions_procesed.pop('V5')
    transactions_procesed.pop('V13')
    transactions_procesed.pop('V27')
    transactions_procesed.pop('V24')
    return transactions_procesed
 

def predict(transactions):
    # load model
    with open('nb_tmp_model', 'rb') as f:
        rfc = pickle.load(f)

    y_pred= rfc.predict(preprocess(transactions))

    return y_pred
