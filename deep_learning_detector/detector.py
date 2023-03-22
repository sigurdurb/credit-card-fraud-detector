import pandas as pd
import numpy as np
import tensorflow as tf

from scipy.stats import reciprocal
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras

MSE_FRAUD_WEIGHT = 1.0
MSE_NOT_FRAUD_WEIGHT = 0.01


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    credidcard_fraud_df = pd.read_csv('../creditcardfraud/creditcard.csv')

    y = credidcard_fraud_df['Class']
    X = credidcard_fraud_df
    X.pop('Class')

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=True,
                                                                random_state=37, stratify=y)
    return X_train, X_validate, y_train, y_validate


def add_time_of_day(df: pd.DataFrame):
    df["Time-Minute"] = df["Time"] // 60 % (24 * 60)


def weighted_mse(y_true, y_pred):
    weight = tf.where(tf.equal(y_true, 0), MSE_NOT_FRAUD_WEIGHT, MSE_FRAUD_WEIGHT)

    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(tf.square(y_true - y_pred) * weight, axis=-1)


def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = keras.backend.sum(neg_y_true * y_pred)
    tn = keras.backend.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + keras.backend.epsilon())
    return specificity


def sensitivity(y_true, y_pred):
    neg_y_pred = 1 - y_pred
    tp = keras.backend.sum(y_true * y_pred)
    fn = keras.backend.sum(y_true * neg_y_pred)
    sensitivity = tp / (tp + fn + keras.backend.epsilon())
    return sensitivity


def build_model(input_layer_size=30, hidden_layer_count=3, hidden_size=80, learning_rate=0.05, normalize_data=True) -> keras.Model:
    input = keras.Input(shape=(input_layer_size,))
    if normalize_data:
        x = keras.layers.Normalization()(input)
    else:
        x = input
    for i in range(hidden_layer_count):
        x = keras.layers.Dense(hidden_size, activation='relu')(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)

    detector_model = keras.Model(input, output, name='dl-detector')
    detector_model.compile(metrics=[keras.metrics.Precision(), specificity, sensitivity], loss=weighted_mse,
                           optimizer=keras.optimizers.SGD(learning_rate=learning_rate))

    return detector_model


def run_hypertuning(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    params = {
        'hidden_layer_count': [2, 3, 4, 5, 6],
        'hidden_size': np.arange(40, 160),
        'learning_rate': reciprocal(0.005, 0.08),
        'normalize_data': [True, False]
    }

    keras_classifier = keras.wrappers.scikit_learn.KerasClassifier(build_model)
    random_grid_search = RandomizedSearchCV(keras_classifier, params, n_iter=300, cv=3, refit='specificity',
                                            scoring={'precision': keras.metrics.Precision(),
                                                     'specificity': make_scorer(specificity),
                                                     'sensitivity': make_scorer(sensitivity)})
    random_grid_search.fit(X_train, y_train, epochs=100, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=10)])

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


def run_detector() -> None:
    X_train, _, y_train, _ = load_data()

    add_time_of_day(X_train)
    X_train.pop('Time')

    run_hypertuning(X_train, y_train)
    # detector_model = build_model(len(X_train.columns), 3, 80, 0.05)


if __name__ == '__main__':
    run_detector()
