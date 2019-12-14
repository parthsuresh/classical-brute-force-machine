import os
import pickle

from sklearn.linear_model import LinearRegression
import pandas as pd
import xlsxwriter

from metrics.regression_metrics import rmse
from metrics.regression_metrics import mae
from metrics.regression_metrics import r2
from metrics.regression_metrics import pearson_correlation

class LinearRegressionModel():
    def __init__(self, X_train, y_train, n_runs, results_path):
        os.mkdir(results_path+'/linear_models')
        print("Training Linear Regression Model....")
        for n in range(n_runs):
            lr_model = LinearRegression().fit(X_train, y_train)
            filename = results_path + '/linear_models/linear_model_' + str(n) + '.sav'
            pickle.dump(lr_model, open(filename, 'wb'))
        print("Training Completed.")

    @classmethod
    def predict(self, X_test, results_path, model_name='best'):
        filename = results_path + '/linear_models/linear_model_'+str(model_name)+'.sav'
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

        workbook = xlsxwriter.Workbook(models_scores_path+'linear_regression_results.xlsx')
        worksheet = workbook.add_worksheet()

        row, column = 0, 0
        worksheet.write(row, column, "Model Name")

        f = open(models_scores_path+"metric.txt", "a")
        for n in range(n_runs):
            model_path = results_path + '/linear_models/linear_model_'+str(n)+'.sav'
            preds = self.predict(X_test, results_path, n)
            f.write("Linear Regression Model " + str(n)+ "\t")
            row = n + 1
            worksheet.write(row, 0, "Linear Regression Model " + str(n)+ "\t")

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

        filename = results_path + '/linear_models/linear_model_best.sav'
        pickle.dump(best_model, open(filename, 'wb'))
        workbook.close()
