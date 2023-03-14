import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    credidcard_fraud_df = pd.read_csv('../creditcardfraud/creditcard.csv')

    y = credidcard_fraud_df['Class']
    X = credidcard_fraud_df
    X.pop('Class')

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=True, random_state=37, stratify=y)
    return X_train, X_validate, y_train, y_validate


def run_detector() -> None:
    X_train, _, y_train, _ = load_data()

    X_train.pop('Time')
    print()

    input = keras.Input(shape=(len(X_train.columns),))
    x = layers.Dense(80)(input)
    x = layers.Dense(80)(x)
    x = layers.Dense(80)(x)
    output = layers.Dense(1)(x)

    detector_model = keras.Model(input, output, name='dl-detector')
    # detector_model.compile()


if __name__ == '__main__':
    run_detector()
