import os
import pickle

from sklearn.linear_model import LogisticRegression
import pandas as pd

from metrics.classification_metrics import f1
from metrics.classification_metrics import roc
from metrics.classification_metrics import auc
from metrics.classification_metrics import accuracy

class LogisticRegressionModel():
    def __init__(self, X_train, y_train, results_path):
        print("Training Logistic Regression Model....")
        lr_model = LogisticRegression().fit(X_train, y_train)
        filename = results_path + '/logistic_model.sav'
        pickle.dump(lr_model, open(filename, 'wb'))
        print("Training Completed.")

    @classmethod
    def predict(self, X_test, results_path):
        filename = results_path + '/logistic_model.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        return loaded_model.predict(X_test)

    def feature_importances(results_path):
        filename = results_path + '/logistic_model.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        return loaded_model.coef_

    @classmethod
    def record_scores(self, X_test, y_test, metrics, results_path):
        models_scores_path = results_path + '/model_scores/'
        preds = self.predict(X_test, results_path)
        f = open(models_scores_path+"metric.txt", "a")
        f.write("Logistic Regression\t")
        if metrics['f1']:
            f1_sc = f1(y_test, preds)
            f.write("F1 score : " + str(f1) + "\t")
        if metrics['accuracy']:
            acc = accuracy(y_test, preds)
            f.write("Accuracy : " + str(acc) + "\t")
        if metrics['roc']:
            roc_curve = roc(y_test, preds)
            f.write("ROC : " + str(roc_curve) + "\t")
        if metrics['auc']:
            auroc = auc(y_test, preds)
            f.write("Area under ROC : " + str(auroc) + "\t")
        f.write("\n")
        f.close()
