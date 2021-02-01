# train-validation-test split

def train_test_val_slpit(df):
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))