from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score


def f1(y_preds, y_actual):
    return f1_score(y_actual, y_preds)


def roc(y_preds, y_actual):
    return roc_curve(y_actual, y_preds)


def auc(y_preds, y_actual):
    return roc_auc_score(y_actual, y_preds)


def accuracy(y_preds, y_actual):
    return accuracy_score(y_preds, y_actual)

def balanced_accuracy(y_preds, y_actual):
    return balanced_accuracy_score(y_actual, y_preds)
