import os
import pickle

from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

from metrics.classification_metrics import f1
from metrics.classification_metrics import roc
from metrics.classification_metrics import auc
from metrics.classification_metrics import accuracy

class ExtraTreesClassificationModel():
    def __init__(self, X_train, y_train, X_val, y_val, ext_params, results_path):
        print("Training Extra Trees Classification Model...")
        # Get params
        n_estimators_list = ext_params['n_estimators']
        min_samples_split_list = ext_params['min_samples_split']
        min_samples_leaf_list = ext_params['min_samples_leaf']
        max_depth_list = ext_params['max_depth']
        max_features = ext_params['max_features']
        model_selection_metric = ext_params['model_selection_metric']

        best_f1 = 0
        best_roc = 0
        best_auc = 0
        best_acc = 0
        best_model = None

        for n_estimators in n_estimators_list:
            for min_samples_split in min_samples_split_list:
                for min_samples_leaf in min_samples_leaf_list:
                    for max_depth in max_depth_list:
                        ext_estimator = ExtraTreesClassifier(n_estimators=n_estimators,
                                                                    min_samples_split=min_samples_split,
                                                                    min_samples_leaf=min_samples_leaf,
                                                                    max_depth=max_depth,
                                                                    max_features=max_features)
                        ext_estimator.fit(X_train, y_train)
                        preds = ext_estimator.predict(X_val)

                        if model_selection_metric == "f1":
                            f = f1(y_val, preds)
                            best_f1, best_model =  (f, ext_estimator)  if f > best_f1 else (best_f1, best_model)
                        elif model_selection_metric == "roc":
                            r = roc(y_val, preds)
                            best_roc, best_model = (r, ext_estimator) if r > best_roc else (best_roc, best_model)
                        elif model_selection_metric == "auc":
                            au = auc(y_val, preds)
                            best_auc, best_model = (auc, ext_estimator) if auc > best_auc else (best_auc, best_model)
                        elif model_selection_metric == "accuracy":
                            acc = accuracy(preds, y_val)
                            best_acc, best_model = (acc, ext_estimator) if acc > best_acc else (best_acc, best_model)
                        else:
                            print("Wrong model selection metric entered!")


        self.ext_model = best_model
        filename = results_path + '/ext_model.sav'
        pickle.dump(self.ext_model, open(filename, 'wb'))
        print("Training Extra Trees Classification Model completed.")

    @classmethod
    def predict(self, X_test, results_path):
        filename = results_path + '/ext_model.sav'
        loaded_model = pickle.load(open(filename,'rb'))
        return loaded_model.predict(X_test)

    @classmethod
    def record_scores(self, X_test, y_test, metrics, results_path):
        models_scores_path = results_path + '/model_scores/'
        preds = self.predict(X_test, results_path)
        f = open(models_scores_path+"metric.txt", "a")
        f.write("Extra Trees Classification\t")
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
