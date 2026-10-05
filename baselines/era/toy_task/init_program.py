def fit_predict(x_train, y_train, x_test):
    # Baseline: predict the training mean everywhere.
    mean = sum(y_train) / len(y_train)
    return [mean for _ in x_test]
