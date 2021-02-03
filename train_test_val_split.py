"""train-validation-test split"""

from sklearn.model_selection import train_test_split
from constants import test_ratio, train_ratio, validation_ratio


def data_split(X, y):
    """
    :param X: dataframe with features for the model
    :param y: labels
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio))
    return X_train, X_val, X_test, y_train, y_val, y_test
