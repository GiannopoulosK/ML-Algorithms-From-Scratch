import numpy as np

def calculate_mean(values):
    return sum(values) / len(values)

def calculate_sse(y_test, y_pred):
    if len(y_test) != len(y_pred):
        raise ValueError("y_test and y_pred must be of the same length")
    else:
        SSE = sum((y_test - y_pred) ** 2) / len(y_test)
    return SSE


def calculate_ssto(y_test, y_pred):
    if len(y_test) != len(y_pred):
        raise ValueError("y_test and y_pred must be of the same length")
    else:
        y_mean = calculate_mean(y_test)
        SSTO = sum((y_test - y_mean) ** 2) / len(y_test)
    return SSTO

def r2_score(y_test, y_pred):
    SSE = calculate_sse(y_test, y_pred)
    SSTO = calculate_ssto(y_test, y_pred)
    r2_score = 1 - (SSE / SSTO)
    return r2_score

def MSE(y_test, y_pred):
    SSE = calculate_sse(y_test, y_pred)
    MSE = SSE / len(y_test)
    return MSE

def RMSE(y_test, y_pred):
    SSE = calculate_sse(y_test, y_pred)
    RMSE = np.sqrt(SSE / len(y_test))
    return RMSE

def MAE(y_test, y_pred):
    MAE = sum(abs(yi_test - yi_pred) for yi_test, yi_pred in zip(y_test, y_pred)) / len(y_test)
    return MAE