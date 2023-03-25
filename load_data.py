import pandas as pd
from sklearn.model_selection import train_test_split


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    credidcard_fraud_df = pd.read_csv('../creditcardfraud/creditcard.csv')

    y = credidcard_fraud_df['Class']
    X = credidcard_fraud_df
    X.pop('Class')

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=True,
                                                                random_state=37, stratify=y)
    return X_train, X_validate, y_train, y_validate