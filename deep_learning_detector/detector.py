import sys

sys.path.append('../')

import pandas as pd
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from tensorflow import keras

MSE_FRAUD_WEIGHT = 1.0
MSE_NOT_FRAUD_WEIGHT = 0.018


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        ['V2', 'V3', 'V4', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V26', 'V27']]


def create_weighted_mse(fraud_weight=1.0, not_fraud_weight=0.05) -> callable:
    def weighted_mse(y_true, y_pred):
        weight = tf.where(tf.equal(y_true, 0), not_fraud_weight, fraud_weight)

        y_true = tf.cast(y_true, y_pred.dtype)
        weight = tf.cast(weight, y_pred.dtype)
        return tf.reduce_mean(tf.square(y_true - y_pred) * weight, axis=-1)

    return weighted_mse


def build_model(input_layer_size=30, hidden_layer_count=5, hidden_size=80, learning_rate=0.05, normalize_data=False,
                not_fraud_mse_weight=0.05) -> keras.Model:
    input = keras.Input(shape=(input_layer_size,), batch_size=100)
    if normalize_data:
        x = keras.layers.Normalization()(input)
    else:
        x = input
    for i in range(hidden_layer_count):
        x = keras.layers.Dense(hidden_size, activation='relu')(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)

    detector_model = keras.Model(input, output, name='dl-detector')
    detector_model.compile(metrics=[keras.metrics.Precision(), keras.metrics.Recall()],
                           loss=create_weighted_mse(not_fraud_weight=not_fraud_mse_weight),
                           optimizer=keras.optimizers.SGD(learning_rate=learning_rate))

    return detector_model


def run_hypertuning(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    params = {
        # 'not_fraud_mse_weight': np.arange(0.001, 0.1, 0.001),
        'hidden_layer_count': [3, 4, 5],
        'hidden_size': [40, 80, 120],
        'learning_rate': [0.01, 0.05, 0.08],
    }

    keras_classifier = keras.wrappers.scikit_learn.KerasClassifier(build_model)
    random_grid_search = GridSearchCV(keras_classifier, params, cv=3, refit='precision',
                                      scoring=['precision', 'recall'])
    random_grid_search.fit(X_train, y_train, epochs=100,
                           callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=10)])

    print(random_grid_search.best_params_)
    print(random_grid_search.best_score_)

    with open('hypertuning-results.txt', 'w') as f:
        f.write('Hyper-parameter tuning\n')
        f.write(f'{random_grid_search.best_params_}\n')
        f.write(f'{random_grid_search.best_score_}\n')

    y_pred = random_grid_search.predict(X_train)
    cm = confusion_matrix(y_train, y_pred)

    print(cm)

    with open('hypertuning-results-cm.txt', 'w') as f:
        f.write('Hyper-parameter confusion matrix\n')
        f.write(f'{str(cm)}\n')


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                save_as: str | None = None) -> keras.wrappers.scikit_learn.KerasClassifier:
    classifier = keras.wrappers.scikit_learn.KerasClassifier(build_model,
                                                             input_layer_size=len(X_train.columns),
                                                             hidden_layer_count=6,
                                                             hidden_size=160,
                                                             learning_rate=0.05,
                                                             not_fraud_mse_weight=MSE_NOT_FRAUD_WEIGHT,
                                                             normalize_data=False)
    classifier.fit(X_train, y_train, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=10)])

    if save_as is not None:
        classifier.model.save(save_as)

    return classifier


def load_model(path: str) -> keras.wrappers.scikit_learn.KerasClassifier:
    model = keras.models.load_model(path, custom_objects={
        'weighted_mse': create_weighted_mse(not_fraud_weight=MSE_NOT_FRAUD_WEIGHT)})

    return model


def predict(transactions: pd.DataFrame) -> pd.Series:
    model = load_model('best_model.h5')
    return ((model.predict(preprocess_data(transactions)) > 0.5).astype("int32")).flatten()
