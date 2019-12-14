import os
import pickle

from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

from metrics.regression_metrics import rmse
from metrics.regression_metrics import mae
from metrics.regression_metrics import r2
from metrics.regression_metrics import pearson_correlation


class GradientBoostingRegressorModel():
    def __init__(self, X_train, y_train, X_val, y_val, gbm_params, results_path):
        print("Training GBM Model...")
        # Get params
        learning_rate_list = gbm_params['learning_rate']
        n_estimators_list = gbm_params['n_estimators']
        min_samples_split_list = gbm_params['min_samples_split']
        min_samples_leaf_list = gbm_params['min_samples_leaf']
        max_depth_list = gbm_params['max_depth']
        max_features = gbm_params['max_features']
        subsample_list = gbm_params['subsample']
        model_selection_metric = gbm_params['model_selection_metric']

        best_me = float('inf')
        best_rms = float('inf')
        best_rsq = 0
        best_pcorr = 0
        best_model = None

        for n in range(n_runs):

            best_me_run = float('inf')
            best_rms_run = float('inf')
            best_rsq_run = 0
            best_pcorr_run = 0
            best_model_run = None

            for learning_rate in learning_rate_list:
                for n_estimators in n_estimators_list:
                    for min_samples_split in min_samples_split_list:
                        for min_samples_leaf in min_samples_leaf_list:
                            for max_depth in max_depth_list:
                                for subsample in subsample_list:
                                    gbm_estimator = GradientBoostingRegressor(learning_rate=learning_rate,
                                                                                n_estimators=n_estimators,
                                                                                min_samples_split=min_samples_split,
                                                                                min_samples_leaf=min_samples_leaf,
                                                                                max_depth=max_depth,
                                                                                subsample=subsample,
                                                                                max_features=max_features)
                                    gbm_estimator.fit(X_train, y_train)
                                    preds = gbm_estimator.predict(X_val)

                                    if model_selection_metric == "mae":
                                        me = mae(preds, y_val)
                                        best_me_run, best_model_run =  (me, gbm_estimator)  if me < best_me_run else (best_me_run, best_model_run)
                                        best_me, best_model =  (me, gbm_estimator)  if me < best_me else (best_me, best_model)
                                    elif model_selection_metric == "rmse":
                                        rms = rmse(preds, y_val)
                                        best_rms, best_model = (rms, gbm_estimator) if rms < best_rms else (best_rms, best_model)
                                    elif model_selection_metric == "r_squared":
                                        rsq = r2(preds, y_val)
                                        best_rsq, best_model = (rsq, gbm_estimator) if rsq > best_rsq else (best_rsq, best_model)
                                    elif model_selection_metric == "pearson_correlation":
                                        pcorr = pearson_correlation(preds, y_val)
                                        best_pcorr, best_model = (pcorr, gbm_estimator) if pcorr > best_pcorr else (best_pcorr, best_model)
                                    else:
                                        print("Wrong model selection metric entered!")


        self.gbm_model = best_model
        filename = results_path + '/gbm_model.sav'
        pickle.dump(self.gbm_model, open(filename, 'wb'))
        print("Training GBM Model completed.")

    @classmethod
    def predict(self, X_test, results_path):
        filename = results_path + '/gbm_model.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        return loaded_model.predict(X_test)

    @classmethod
    def record_scores(self, X_test, y_test, metrics, results_path):
        models_scores_path = results_path + '/model_scores/'
        preds = self.predict(X_test, results_path)
        f = open(models_scores_path+"metric.txt", "a")
        f.write("GBM Regression\t")
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
