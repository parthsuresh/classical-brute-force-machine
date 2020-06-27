import os
import pickle

from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import xlsxwriter

from metrics.regression_metrics import rmse
from metrics.regression_metrics import mae
from metrics.regression_metrics import r2
from metrics.regression_metrics import pearson_correlation


class GradientBoostingRegressorModel():
    def __init__(self, X_train, y_train, X_val, y_val, gbm_params, n_runs, results_path):
        print("Training GBM Model...")
        os.mkdir(results_path+'/gbm_models')
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
        best_model_params = {}

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
                                    model_params = {
                                        'learning_rate':learning_rate,
                                        'n_estimators':n_estimators,
                                        'min_samples_split':min_samples_split,
                                        'min_samples_leaf':min_samples_leaf,
                                        'max_depth':max_depth,
                                        'subsample':subsample,
                                        'max_features':max_features
                                    }
                                    gbm_estimator.fit(X_train, y_train)
                                    preds = gbm_estimator.predict(X_val)

                                    if model_selection_metric == "mae":
                                        me = mae(preds, y_val)
                                        best_me_run, best_model_run =  (me, gbm_estimator)  if me < best_me_run else (best_me_run, best_model_run)
                                        best_me, best_model, best_model_params =  (me, gbm_estimator, model_params)  if me < best_me else (best_me, best_model, best_model_params)
                                    elif model_selection_metric == "rmse":
                                        rms = rmse(preds, y_val)
                                        best_rms_run, best_model_run = (rms, gbm_estimator) if rms < best_rms_run else (best_rms_run, best_model_run)
                                        best_rms, best_model, best_model_params = (rms, gbm_estimator. model_params) if rms < best_rms else (best_rms, best_model, best_model_params)
                                    elif model_selection_metric == "r_squared":
                                        rsq = r2(preds, y_val)
                                        best_rsq_run, best_model_run = (rsq, gbm_estimator) if rsq > best_rsq_run else (best_rsq_run, best_model_run)
                                        best_rsq, best_model, best_model_params = (rsq, gbm_estimator, model_params) if rsq > best_rsq else (best_rsq, best_model, best_model_params)
                                    elif model_selection_metric == "pearson_correlation":
                                        pcorr, _ = pearson_correlation(preds, y_val)
                                        best_pcorr_run, best_model_run = (pcorr, gbm_estimator) if pcorr > best_pcorr_run else (best_pcorr_run, best_model_run)
                                        best_pcorr, best_model, best_model_params = (pcorr, gbm_estimator, model_params) if pcorr > best_pcorr else (best_pcorr, best_model, best_model_params)
                                    else:
                                        print("Wrong model selection metric entered!")

            filename = results_path + '/gbm_models/gbm_model_' + str(n) + '.sav'
            pickle.dump(best_model_run, open(filename, 'wb'))

        self.gbm_model = best_model
        filename = results_path + '/gbm_models/gbm_model.sav'
        pickle.dump(self.gbm_model, open(filename, 'wb'))
        f = open(results_path+'/gbm_models/best_scores.txt', 'w')
        f.write(str(best_model_params))
        f.close()      
        print("Training GBM Model completed.")

    @classmethod
    def predict(self, X_test, results_path, model_name='best'):
        filename = results_path + '/gbm_models/gbm_model_'+str(model_name)+'.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        return loaded_model.predict(X_test)

    @classmethod
    def record_scores(self, X_test, y_test, metrics, n_runs, results_path):
        models_scores_path = results_path + '/model_scores/'

        best_rmse = 0
        best_mae = 0
        best_r2 = 0
        best_corr = 0
        best_model = None

        workbook = xlsxwriter.Workbook(models_scores_path+'gbm_results.xlsx')
        worksheet = workbook.add_worksheet()

        row, column = 0, 0
        worksheet.write(row, column, "Model Name")

        f = open(models_scores_path+"metric.txt", "a")
        for n in range(n_runs):
            model_path = results_path + '/gbm_models/gbm_model_'+str(n)+'.sav'
            preds = self.predict(X_test, results_path, n)
            f.write("GBM Model " + str(n)+ "\t")
            row = n + 1
            worksheet.write(row, 0, "GBM Model " + str(n)+ "\t")

            column = 0

            if metrics['rmse']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "RMSE")
                rmse_sc = rmse(y_test, preds)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_rmse, best_model =  (rmse_sc, model)  if rmse_sc > best_rmse else (best_rmse, best_model)
                f.write("RMSE : " + str(rmse_sc) + "\t")
                worksheet.write(row, column, rmse_sc)
            if metrics['mae']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "MAE")
                me = mae(y_test, preds)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_mae, best_model =  (me, model)  if me > best_mae else (best_mae, best_model)
                f.write("MAE : " + str(me) + "\t")
                worksheet.write(row, column, me)
            if metrics['r_squared']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "R^2")
                rsq = r2(y_test, preds)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_r2, best_model =  (rsq, model)  if rsq > best_r2 else (best_r2, best_model)
                f.write("R^2 : " + str(rsq) + "\t")
                worksheet.write(row, column, rsq)
            if metrics['pearson_correlation']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "Pearson Correlation")
                pcorr, _ = pearson_correlation(y_test, preds)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_corr, best_model =  (pcorr, model)  if pcorr > best_corr else (best_corr, best_model)
                f.write("Pearson Correlation : " + str(pcorr) + "\t")
                worksheet.write(row, column, pcorr)
            f.write("\n")
        f.close()

        filename = results_path + '/gbm_models/gbm_model_best.sav'
        pickle.dump(best_model, open(filename, 'wb'))
        workbook.close()
