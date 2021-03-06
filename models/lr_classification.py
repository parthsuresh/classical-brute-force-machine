import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import xlsxwriter
import shap
import matplotlib.pyplot as plt

from metrics.classification_metrics import f1
from metrics.classification_metrics import roc
from metrics.classification_metrics import auc
from metrics.classification_metrics import accuracy

class LogisticRegressionModel():
    def __init__(self, X_train, y_train, n_runs, results_path):
        self.explainer = None
        y_train = np.squeeze(np.array(y_train))
        os.mkdir(results_path+'/logistic_models')
        print("Training Logistic Regression Model....")
        for n in range(n_runs):
            lr_model = LogisticRegression()
            lr_model.fit(X_train, y_train)
            filename = results_path + '/logistic_models/logistic_model_' + str(n) + '.sav'
            pickle.dump(lr_model, open(filename, 'wb'))
        print("Training Completed.")
        self.explainer = shap.LinearExplainer(lr_model, X_train)


    def predict(self, X_test, results_path, model_name='best'):
        filename = results_path + '/logistic_models/logistic_model_'+str(model_name)+'.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        return loaded_model.predict(X_test)

    def feature_importances(self, results_path, feature_names):
        filename = results_path + '/logistic_models/logistic_model.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        importances = np.squeeze(loaded_model.coef_)
        indices = np.argsort(np.abs(importances))
        features = list(feature_names)
        plots_path = results_path + '/logistic_plots/'
        if not os.path.exists(plots_path):
            os.mkdir(plots_path)
        f = open(plots_path + "lr_feature_importances.txt", "w")
        f.write("Feature Importances\n")
        for i in reversed(indices):
            f.write(str(features[i]) + " : "  + str(importances[i]) + "\n")
        f.close()

    def record_scores(self, X_test, y_test, n_runs, metrics, feature_names, max_display_features, results_path):
        models_scores_path = results_path + '/model_scores/'

        best_f1 = 0
        best_roc = 0
        best_auc = 0
        best_acc = 0
        best_model = None

        workbook = xlsxwriter.Workbook(models_scores_path+'logistic_regression_results.xlsx')
        worksheet = workbook.add_worksheet()

        row, column = 0, 0
        worksheet.write(row, column, "Model Name")

        f = open(models_scores_path+"metric.txt", "a")
        for n in range(n_runs):
            model_path = results_path + '/logistic_models/logistic_model_'+str(n)+'.sav'
            preds = self.predict(X_test, results_path, n)
            f.write("Logistic Regression Model " + str(n)+ "\t")
            row = n + 1
            worksheet.write(row, 0, "Logistic Regression Model " + str(n)+ "\t")

            column = 0

            if metrics['f1']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "F1")
                f1_sc = f1(y_test, preds)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_f1, best_model =  (f1_sc, model)  if f1_sc > best_f1 else (best_f1, best_model)
                f.write("F1 score : " + str(f1_sc) + "\t")
                worksheet.write(row, column, f1_sc)
            if metrics['accuracy']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "Accuracy")
                acc = accuracy(preds, y_test)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_acc, best_model =  (acc, model)  if acc > best_acc else (best_acc, best_model)
                f.write("Accuracy : " + str(acc) + "\t")
                worksheet.write(row, column, acc)
            if metrics['balanced_accuracy']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "Balanced Accuracy")
                bal_acc = balanced_accuracy(y_preds, y_true)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_bal_acc, best_model = (bal_acc, model) if bal_acc > best_acc else (best_bal_acc, best_model)
                f.write("Balanced Accuracy : " + str(bal_acc) + "\t")
                worksheet.write(row, column, bal_acc)
            if metrics['roc']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "ROC")
                roc_curve = roc(y_test, preds)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_roc, best_model =  (roc_curve, model)  if roc_curve > best_roc else (best_roc, best_model)
                f.write("ROC : " + str(roc_curve) + "\t")
                worksheet.write(row, column, roc_curve)
            if metrics['auc']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "AUC")
                auroc = auc(y_test, preds)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_auc, best_model =  (auroc, model)  if auroc > best_auc else (best_auc, best_model)
                f.write("Area under ROC : " + str(auroc) + "\t")
                worksheet.write(row, column, auroc)
            f.write("\n")
        f.close()

        filename = results_path + '/logistic_models/logistic_model.sav'
        pickle.dump(best_model, open(filename, 'wb'))
        self.feature_importances(results_path, feature_names.values)
        shap_values = self.explainer.shap_values(X_test)
        X_test_array = np.array(X_test)
        shap.summary_plot(shap_values, X_test, feature_names = feature_names, show=False, plot_type="bar", max_display=max_display_features)
        plt.savefig(results_path + '/logistic_plots/features.png',  bbox_inches='tight')
        plt.close()
        shap.summary_plot(shap_values, X_test, feature_names = feature_names, show=False, max_display=max_display_features)
        plt.savefig(results_path + '/logistic_plots/features_summary.png',  bbox_inches='tight')
        plt.close()
        vals= np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(X_test.columns, vals)), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        feature_importance.to_csv(index=False, path_or_buf=results_path+"/logistic_plots/feature_importances.csv")
        workbook.close()
