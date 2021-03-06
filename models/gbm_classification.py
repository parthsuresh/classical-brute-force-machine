import os
import pickle

from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import xlsxwriter
import numpy as np
import matplotlib.pyplot as plt
import shap

from metrics.classification_metrics import f1
from metrics.classification_metrics import roc
from metrics.classification_metrics import auc
from metrics.classification_metrics import accuracy
from metrics.classification_metrics import balanced_accuracy


class GradientBoostingClassificationModel:
    def __init__(self, x_train, y_train, x_val, y_val, gbm_params, n_runs, results_path):
        self.explainer = None
        y_train = np.squeeze(np.array(y_train))
        y_val = np.squeeze(np.array(y_val))
        print("Training GBM Model...")
        if not os.path.exists(results_path + '/gbm_models'):
            os.mkdir(results_path + '/gbm_models')
        # Get params
        learning_rate_list = gbm_params['learning_rate']
        n_estimators_list = gbm_params['n_estimators']
        min_samples_split_list = gbm_params['min_samples_split']
        min_samples_leaf_list = gbm_params['min_samples_leaf']
        max_depth_list = gbm_params['max_depth']
        max_features = gbm_params['max_features']
        subsample_list = gbm_params['subsample']
        model_selection_metric = gbm_params['model_selection_metric']

        best_f1 = 0
        best_roc = 0
        best_auc = 0
        best_acc = 0
        best_bal_acc = 0
        best_model = None
        best_model_params = {}

        for n in range(n_runs):

            best_f1_run = 0
            best_roc_run = 0
            best_auc_run = 0
            best_acc_run = 0
            best_bal_acc_run = 0
            best_model_run = None
            

            for learning_rate in learning_rate_list:
                for n_estimators in n_estimators_list:
                    for min_samples_split in min_samples_split_list:
                        for min_samples_leaf in min_samples_leaf_list:
                            for max_depth in max_depth_list:
                                for subsample in subsample_list:
                                    gbm_estimator = GradientBoostingClassifier(learning_rate=learning_rate,
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
                                    gbm_estimator.fit(x_train, y_train)
                                    preds = gbm_estimator.predict(x_val)

                                    if model_selection_metric == "f1":
                                        f = f1(y_val, preds)
                                        best_f1_run, best_model_run = (f, gbm_estimator) if f > best_f1_run else (
                                            best_f1_run, best_model_run)
                                        best_f1, best_model, best_model_params = (f, gbm_estimator, model_params) if f > best_f1 else (
                                            best_f1, best_model, best_model_params)
                                    elif model_selection_metric == "roc":
                                        r = roc(y_val, preds)
                                        best_roc_run, best_model_run = (r, gbm_estimator) if r > best_roc_run else (
                                            best_roc_run, best_model_run)
                                        best_roc, best_model, best_model_params = (r, gbm_estimator, model_params) if r > best_roc else (
                                            best_roc, best_model, best_model_params)
                                    elif model_selection_metric == "auc":
                                        au = auc(y_val, preds)
                                        best_auc_run, best_model_run = (au, gbm_estimator) if au > best_auc_run else (
                                            best_auc_run, best_model_run)
                                        best_auc, best_model, best_model_params = (au, gbm_estimator, model_params) if au > best_auc else (
                                            best_auc, best_model, best_model_params)
                                    elif model_selection_metric == "accuracy":
                                        acc = accuracy(preds, y_val)
                                        best_acc_run, best_model_run = (acc, gbm_estimator) if acc > best_acc_run else (
                                            best_acc_run, best_model_run)
                                        best_acc, best_model, best_model_params = (acc, gbm_estimator, model_params) if acc > best_acc else (
                                            best_acc, best_model, best_model_params)
                                    elif model_selection_metric == "balanced_accuracy":
                                        bal_acc = balanced_accuracy(preds, y_val)
                                        best_bal_acc_run, best_model_run = (bal_acc, gbm_estimator) if bal_acc > best_bal_acc_run else (
                                            best_bal_acc_run, best_model_run)
                                        best_bal_acc, best_model, best_model_params = (bal_acc, gbm_estimator, model_params) if bal_acc > best_bal_acc else (
                                            best_bal_acc, best_model, best_model_params)
                                    else:
                                        print("Wrong model selection metric entered!")

            filename = results_path + '/gbm_models/gbm_model_' + str(n) + '.sav'
            pickle.dump(best_model_run, open(filename, 'wb'))

        self.gbm_model = best_model
        filename = results_path + '/gbm_models/gbm_model.sav'
        pickle.dump(self.gbm_model, open(filename, 'wb'))
        f = open(results_path+'/gbm_models/best_params.txt', 'w')
        f.write(str(best_model_params))
        f.close()
        self.explainer = shap.TreeExplainer(best_model, x_train)

    def predict(self, X_test, results_path, model_name='best'):
        filename = results_path + '/gbm_models/gbm_model_' + str(model_name) + '.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model.predict(X_test)

    def feature_importances(self, results_path, feature_names):
        plots_path = results_path + '/gbm_class_plots/'
        if not os.path.exists(plots_path):
            os.mkdir(plots_path)
        filename = results_path + '/gbm_models/gbm_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        features, importances = list(feature_names), loaded_model.feature_importances_
        indices = np.argsort(importances)

        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.savefig(plots_path + 'random_forest_feature_importances.png')

        f = open(plots_path + "gbm_feature_importances.txt", "w")
        f.write("Feature Importances\n")
        for i in reversed(indices):
            f.write(str(features[i]) + " : " + str(importances[i]) + "\n")
        f.close()
        # self.ext_model = best_model
        # filename = results_path + '/random_forest_models/rf_model.sav'
        # pickle.dump(self.ext_model, open(filename, 'wb'))
        print("Training GBM Classification Model completed.")

    def record_scores(self, X_test, y_test, metrics, n_runs, max_display_features, feature_names, results_path):
        models_scores_path = results_path + '/model_scores/'

        best_f1 = 0
        best_acc = 0
        best_roc = 0
        best_auc = 0
        best_bal_acc = 0
        best_model = None

        workbook = xlsxwriter.Workbook(models_scores_path + 'gbm_results.xlsx')
        worksheet = workbook.add_worksheet()

        row, column = 0, 0
        worksheet.write(row, column, "Model Name")

        f = open(models_scores_path + "metric.txt", "a")
        for n in range(n_runs):
            model_path = results_path + '/gbm_models/gbm_model_' + str(n) + '.sav'
            preds = self.predict(X_test, results_path, n)
            f.write("GBM Model " + str(n) + "\t")
            row = n + 1
            worksheet.write(row, 0, "GBM Model " + str(n) + "\t")

            column = 0

            if metrics['f1']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "F1")
                f1_sc = f1(y_test, preds)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_f1, best_model = (f1_sc, model) if f1_sc > best_f1 else (best_f1, best_model)
                f.write("F1 score : " + str(f1_sc) + "\t")
                worksheet.write(row, column, f1_sc)
            if metrics['accuracy']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "Accuracy")
                acc = accuracy( preds, y_test)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_acc, best_model = (acc, model) if acc > best_acc else (best_acc, best_model)
                f.write("Accuracy : " + str(acc) + "\t")
                worksheet.write(row, column, acc)
            if metrics['balanced_accuracy']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "Balanced Accuracy")
                bal_acc = balanced_accuracy(preds, y_test)
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
                best_roc, best_model = (roc_curve, model) if roc_curve > best_roc else (best_roc, best_model)
                f.write("ROC : " + str(roc_curve) + "\t")
                worksheet.write(row, column, roc_curve)
            if metrics['auc']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "AUC")
                auroc = auc(y_test, preds)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_auc, best_model = (auroc, model) if auroc > best_auc else (best_auc, best_model)
                f.write("AUC : " + str(auc) + "\t")
                worksheet.write(row, column, auc)
            f.write("\n")
        f.close()
        workbook.close()
