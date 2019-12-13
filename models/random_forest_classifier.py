import os
import pickle

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import numpy as np

from metrics.classification_metrics import f1
from metrics.classification_metrics import roc
from metrics.classification_metrics import auc
from metrics.classification_metrics import accuracy

class RandomForestClassificationModel():
    def __init__(self, X_train, y_train, X_val, y_val, n_runs, ext_params, results_path):
        print("Training Random Forest Classification Model...")
        os.mkdir(results_path+'/random_forest_models')

        # Get params
        n_estimators_list = ext_params['n_estimators']
        min_samples_split_list = ext_params['min_samples_split']
        min_samples_leaf_list = ext_params['min_samples_leaf']
        max_features = ext_params['max_features']
        model_selection_metric = ext_params['model_selection_metric']

        best_f1 = 0
        best_roc = 0
        best_auc = 0
        best_acc = 0
        best_model = None

        for n in range(n_runs):

            best_f1_run = 0
            best_roc_run = 0
            best_auc_run = 0
            best_acc_run = 0
            best_model_run = None

            for n_estimators in n_estimators_list:
                for min_samples_split in min_samples_split_list:
                    for min_samples_leaf in min_samples_leaf_list:
                            ext_estimator = RandomForestClassifier(n_estimators=n_estimators,
                                                                        min_samples_split=min_samples_split,
                                                                        min_samples_leaf=min_samples_leaf,
                                                                        max_features=max_features)
                            ext_estimator.fit(X_train, y_train)
                            preds = ext_estimator.predict(X_val)

                            if model_selection_metric == "f1":
                                f = f1(y_val, preds)
                                best_f1_run, best_model_run =  (f, ext_estimator)  if f > best_f1_run else (best_f1_run, best_model_run)
                                best_f1, best_model =  (f, ext_estimator)  if f > best_f1 else (best_f1, best_model)
                            elif model_selection_metric == "roc":
                                r = roc(y_val, preds)
                                best_roc_run, best_model_run = (r, ext_estimator) if r > best_roc_run else (best_roc_run, best_model_run)
                                best_roc, best_model = (r, ext_estimator) if r > best_roc else (best_roc, best_model)
                            elif model_selection_metric == "auc":
                                au = auc(y_val, preds)
                                best_auc_run, best_model_run = (auc, ext_estimator) if auc > best_auc_run else (best_auc_run, best_model_run)
                                best_auc, best_model = (auc, ext_estimator) if auc > best_auc else (best_auc, best_model)
                            elif model_selection_metric == "accuracy":
                                acc = accuracy(preds, y_val)
                                best_acc_run, best_model_run = (acc, ext_estimator) if acc > best_acc_run else (best_acc_run, best_model_run)
                                best_acc, best_model = (acc, ext_estimator) if acc > best_acc else (best_acc, best_model)
                            else:
                                print("Wrong model selection metric entered!")

            filename = results_path + '/random_forest_models/rf_model_' + str(n) + '.sav'
            pickle.dump(best_model_run, open(filename, 'wb'))

        self.ext_model = best_model
        filename = results_path + '/random_forest_models/rf_model.sav'
        pickle.dump(self.ext_model, open(filename, 'wb'))
        print("Training Random Forest Classification Model completed.")

    @classmethod
    def predict(self, X_test, results_path, model_name='best'):
        filename = results_path + '/random_forest_models/rf_model_'+str(model_name)+'.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        return loaded_model.predict(X_test)

    def feature_importances(results_path, feature_names):
        plots_path = results_path + '/plots/'
        filename = results_path + '/random_forest_models/rf_model.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        features, importances =  list(feature_names), loaded_model.feature_importances_
        indices = np.argsort(importances)
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.savefig(plots_path + 'random_forest_feature_importances.png')

        f = open(plots_path + "feature_importances.txt", "w")
        f.write("Feature Importances\n")
        for i in reversed(indices):
            f.write(str(features[i]) + " : "  + str(importances[i]) + "\n")
        f.close()



    @classmethod
    def record_scores(self, X_test, y_test, n_runs, metrics, feature_names, results_path):
        models_scores_path = results_path + '/model_scores/'

        workbook = xlsxwriter.Workbook(models_scores_path+'random_forest_results.xlsx')
        worksheet = workbook.add_worksheet()

        row, column = 0, 0
        worksheet.write(row, column, "Model Name")

        f = open(models_scores_path+"metric.txt", "a")

        for n in range(n_runs):
            preds = self.predict(X_test, results_path, model_name=n)
            row = n + 1
            worksheet.write(row, 0, "Random Forest Model " + str(n)+ "\t")
            f.write("Random Forest Classification\t")
            column = 0
            if metrics['f1']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "F1")
                f1_sc = f1(y_test, preds)
                f.write("F1 score : " + str(f1) + "\t")
                worksheet.write(row, column, f1_sc)
            if metrics['accuracy']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "Accuracy")
                acc = accuracy(y_test, preds)
                f.write("Accuracy : " + str(acc) + "\t")
                worksheet.write(row, column, acc)
            if metrics['roc']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "ROC")
                roc_curve = roc(y_test, preds)
                f.write("ROC : " + str(roc_curve) + "\t")
                worksheet.write(row, column, roc_curve)
            if metrics['auc']:
                column += 1
                if n == 0:
                    worksheet.write(0, column, "AUC")
                auroc = auc(y_test, preds)
                f.write("Area under ROC : " + str(auroc) + "\t")
                worksheet.write(row, column, auroc)
            f.write("\n")
        f.close()

        self.feature_importances(results_path, feature_names)

        workbook.close()
