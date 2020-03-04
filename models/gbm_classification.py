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


class GradientBoostingClassificationModel():
    def __init__(self, X_train, y_train, X_val, y_val, gbm_params, n_runs, results_path):
        self.explainer = None
        y_train = np.squeeze(np.array(y_train))
        y_val = np.squeeze(np.array(y_val))
        print("Training GBM Model...")
        if not os.path.exists(results_path+'/gbm_models'):
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
                                    gbm_estimator.fit(X_train, y_train)
                                    preds = gbm_estimator.predict(X_val)

                                    if model_selection_metric == "f1":
                                        f = f1(y_val, preds)
                                        best_f1_run, best_model_run =  (f, gbm_estimator)  if f > best_f1_run else (best_f1_run, best_model_run)
                                        best_f1, best_model=  (f, gbm_estimator)  if f > best_f1 else (best_f1, best_model)
                                    elif model_selection_metric == "roc":
                                        r = roc(y_val, preds)
                                        best_roc_run, best_model_run = (r, gbm_estimator) if r > best_roc_run else (best_roc_run, best_model_run)
                                        best_roc, best_model = (r, gbm_estimator) if r > best_roc else (best_roc, best_model)
                                    elif model_selection_metric == "auc":
                                        au = auc(y_val, preds)
                                        best_auc_run, best_model_run = (au, gbm_estimator) if au > best_auc_run else (best_auc_run, best_model_run)
                                        best_auc, best_model = (au, gbm_estimator) if au > best_auc else (best_auc, best_model)
                                    elif model_selection_metric == "accuracy":
                                        acc = accuracy(preds, y_val)
                                        best_acc_run, best_model_run = (acc, gbm_estimator) if acc > best_acc_run else (best_acc_run, best_model_run)
                                        best_acc, best_model = (acc, gbm_estimator) if acc > best_acc else (best_acc, best_model)
                                    else:
                                        print("Wrong model selection metric entered!")

            filename = results_path + '/gbm_models/gbm_model_' + str(n) + '.sav'
            pickle.dump(best_model_run, open(filename, 'wb'))

        self.gbm_model = best_model
        filename = results_path + '/gbm_models/gbm_model.sav'
        pickle.dump(self.gbm_model, open(filename, 'wb'))
        self.explainer = shap.TreeExplainer(best_model, X_train, feature_dependence="independent")



    def predict(self, X_test, results_path, model_name='best'):
        filename = results_path + '/gbm_models/gbm_model_'+str(model_name)+'.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        return loaded_model.predict(X_test)

    def feature_importances(self, results_path, feature_names):
        plots_path = results_path + '/gbm_class_plots/'
        if not os.path.exists(plots_path):
            os.mkdir(plots_path)
        filename = results_path + '/gbm_models/gbm_model.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        features, importances =  list(feature_names), loaded_model.feature_importances_
        indices = np.argsort(importances)

        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.savefig(plots_path + 'random_forest_feature_importances.png')


        f = open(plots_path + "gbm_feature_importances.txt", "w")
        f.write("Feature Importances\n")
        for i in reversed(indices):
            f.write(str(features[i]) + " : "  + str(importances[i]) + "\n")
        f.close()
        #self.ext_model = best_model
        #filename = results_path + '/random_forest_models/rf_model.sav'
        #pickle.dump(self.ext_model, open(filename, 'wb'))
        print("Training GBM Classification Model completed.")


    def record_scores(self, X_test, y_test, metrics, n_runs, feature_names, results_path):
        models_scores_path = results_path + '/model_scores/'

        best_f1 = 0
        best_acc = 0
        best_roc = 0
        best_auc = 0
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
                acc = accuracy(y_test, preds)
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                best_acc, best_model =  (acc, model)  if acc > best_acc else (best_acc, best_model)
                f.write("Accuracy : " + str(acc) + "\t")
                worksheet.write(row, column, acc)
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
                f.write("AUC : " + str(auc) + "\t")
                worksheet.write(row, column, auc)
            f.write("\n")
        f.close()

        filename = results_path + '/gbm_models/gbm_model.sav'
        pickle.dump(best_model, open(filename, 'wb'))
        self.feature_importances(results_path, feature_names)
        shap_values = self.explainer.shap_values(X_test)
        X_test_array = np.array(X_test)
        shap.summary_plot(shap_values, X_test, feature_names = feature_names, show=False, plot_type="bar")
        plt.savefig(results_path + '/gbm_class_plots/features_bar.png',  bbox_inches='tight')
        plt.close()
        shap.summary_plot(shap_values, X_test, feature_names = feature_names, show=False)
        plt.savefig(results_path + '/gbm_class_plots/features_summary.png',  bbox_inches='tight')
        plt.close()
        vals= np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(X_test.columns, vals)), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        feature_importance.to_csv(index=False, path_or_buf=results_path+"/gbm_class_plots/feature_importances.csv")
        workbook.close()
