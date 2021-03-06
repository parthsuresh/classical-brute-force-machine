from math import sqrt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def rmse(y_preds, y_actual):
    return sqrt(mean_squared_error(y_actual, y_preds))


def mae(y_preds, y_actual):
    return mean_absolute_error(y_actual, y_preds)


def r2(y_preds, y_actual):
    return r2_score(y_actual, y_preds)


def pearson_correlation(y_preds, y_actual):
    return pearsonr(y_actual, y_preds)
