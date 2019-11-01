import os
import pickle

import xgboost as xgb
import pandas as pd

from metrics.regression_metrics import rmse
from metrics.regression_metrics import mae
from metrics.regression_metrics import r2
from metrics.regression_metrics import pearson_correlation


class XGBoostRegressionModel():
    def __init__(self, X_train, y_train, X_val, y_val, xgb_params, results_path):
        print("Training XGBoost Model...")
        # Get params
        learning_rate_list = xgb_params['learning_rate']
        n_estimators_list = xgb_params['n_estimators']
        max_depth_list = xgb_params['max_depth']
        colsample_bytree_list = xgb_params['colsample_bytree']
        gamma_list = xgb_params['gamma']
        alpha_list = xgb_params['gamma']
        lambda_list = xgb_params['lambda']
        subsample_list = xgb_params['subsample']
        model_selection_metric = xgb_params['model_selection_metric']

        best_me = float('inf')
        best_rms = float('inf')
        best_rsq = 0
        best_pcorr = 0
        best_model = None

        for learning_rate in learning_rate_list:
            for n_estimators in n_estimators_list:
                for colsample_bytree in colsample_bytree_list:
                    for gamma in gamma_list:
                        for max_depth in max_depth_list:
                            for subsample in subsample_list:
                                for alpha in alpha_list:
                                    for lamda in lambda_list:
                                        xgb_estimator = xgb.XGBRegressor(objective='reg:squarederror',                                                                          learning_rate=learning_rate,
                                                                        n_estimators=n_estimators,
                                                                        colsample_bytree=colsample_bytree,
                                                                        gamma=gamma,
                                                                        alpha=alpha,
                                                                        reg_lambda=lamda,
                                                                        max_depth=max_depth,
                                                                        subsample=subsample)
                                        xgb_estimator.fit(X_train, y_train)
                                        preds = xgb_estimator.predict(X_val)

                                        if model_selection_metric == "mae":
                                            me = mae(preds, y_val)
                                            best_me, best_model =  (me, xgb_estimator)  if me < best_me else (best_me, best_model)
                                        elif model_selection_metric == "rmse":
                                            rms = rmse(preds, y_val)
                                            best_rms, best_model = (rms, xgb_estimator) if rms < best_rms else (best_rms, best_model)
                                        elif model_selection_metric == "r_squared":
                                            rsq = r2(preds, y_val)
                                            best_rsq, best_model = (rsq, xgb_estimator) if rsq > best_rsq else (best_rsq, best_model)
                                        elif model_selection_metric == "pearson_correlation":
                                            pcorr = pearson_correlation(preds, y_val)
                                            best_pcorr, best_model = (pcorr, xgb_estimator) if pcorr > best_pcorr else (best_pcorr, best_model)
                                        else:
                                            print("Wrong model selection metric entered!")

        self.xgb_model = best_model
        filename = results_path + '/xgb_model.sav'
        pickle.dump(self.xgb_model, open(filename, 'wb'))
        print("Training XGBoost Model completed.")

    @classmethod
    def predict(self, X_test, results_path):
        filename = results_path + '/xgb_model.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        return loaded_model.predict(X_test)

    @classmethod
    def record_scores(self, X_test, y_test, metrics, results_path):
        models_scores_path = results_path + '/model_scores/'
        preds = self.predict(X_test, results_path)
        f = open(models_scores_path+"metric.txt", "a")
        f.write("XGB Regression\t")
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
            f.write("Pearson Correlation : " + str(pcorr[0]) + "\t")
        f.write("\n")
        f.close()
