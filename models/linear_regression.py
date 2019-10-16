import os
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, GridSearchCV
import pandas as pd

from metrics.regression_metrics import rmse
from metrics.regression_metrics import mae
from metrics.regression_metrics import r2
from metrics.regression_metrics import pearson_correlation

class LinearRegressionModel():
    def __init__(self, X_train, y_train, results_path):
        self.lr_model = LinearRegression().fit(X_train, y_train)
        filename = results_path + '/linear_model.sav'
        pickle.dump(self.lr_model, open(filename, 'wb'))

    def get_params(self):
        return self.lr_model.get_params()

    @classmethod
    def predict(self, X_test, results_path):
        filename = results_path + '/linear_model.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        return loaded_model.predict(X_test)

    @classmethod
    def record_scores(self, X_test, y_test, metrics, results_path):
        models_scores_path = results_path + '/model_scores/'
        if not os.path.exists(models_scores_path):
            os.mkdir(models_scores_path)
        preds = self.predict(X_test, results_path)
        f = open(models_scores_path+"metric.txt", "a")
        f.write("Linear Regression\t")
        if metrics['rmse']:
            rms = rmse(preds, y_test)
            f.write("RMSE : " + str(rms) + "\t")
        if metrics['mae']:
            me = mae(preds, y_test)
            f.write("MAE : " + str(me) + "\t")
        if metrics['r_squared']:
            rsq = r2(preds, y_test)
            f.write("R^2 : " + str(rsq) + "\t")
        if metrics['pearson_correlation']:
            pcorr = pearson_correlation(preds, y_test)
            f.write("Pearson Correlation : " + str(pcorr[0][0]) + "\t")
        f.write("\n")
        f.close()
