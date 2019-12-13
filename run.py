import os
import warnings

import argparse
import numpy as np
import pandas as pd

from config_parse import parse_config
from data.data_preprocessing import process_data
from models.linear_regression import LinearRegressionModel
from models.gbm_regression import GradientBoostingRegressorModel
from models.xgboost_regression import XGBoostRegressionModel
from models.lr_classification import LogisticRegressionModel
from models.extra_trees import ExtraTreesClassificationModel
from models.random_forest_classifier import RandomForestClassificationModel

warnings.filterwarnings("ignore", category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='Classical Brute-Force Machine')
    parser.add_argument('--config_path', action='store',
                        required=True,
                        dest='config_path',
                        help='Enter path to TOML file')
    parser.add_argument('--data_path', action='store',
                        required=True,
                        dest='data_path',
                        help='Enter path to dataset')
    parser.add_argument('--output_path', action='store',
                        dest='output_path',
                        help='Enter path to output')
    results = parser.parse_args()
    return results

if __name__ == "__main__":
    args = parse_args()
    args.output_path = args.output_path if args.output_path else "."
    results_path = args.output_path + "/results"
    model_scores_path = results_path + "/model_scores"
    processed_data_path = results_path + "/data"
    plots_path = results_path + "/plots"

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    if not os.path.exists(model_scores_path):
        os.mkdir(model_scores_path)
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)

    config = parse_config(args.config_path)
    X_train, X_val, y_train, y_val, feature_names = process_data(args.data_path, config, args)
    f = open(model_scores_path+"/metric.txt", "a")
    f.close()

    # Models
    if config['common_parameters']['problem_type'] == 'regression':

        if config['regression']['regression_models']['linear_regression']:
            if config['common_parameters']['train']:
                lr = LinearRegressionModel(X_train, y_train, results_path)
            if config['common_parameters']['predict_new']:
                lr = LinearRegressionModel.predict(X_val, results_path)
            if config['common_parameters']['get_results']:
                lr = LinearRegressionModel.record_scores(X_val, y_val, config['regression']['performance_metrics'], results_path)

        if config['regression']['regression_models']['gbm']:
            if config['common_parameters']['train']:
                gbm = GradientBoostingRegressorModel(X_train, y_train, X_val, y_val, config['regression']['regression_models']['gbm_params'], results_path)
            if config['common_parameters']['predict_new']:
                gbm = GradientBoostingRegressorModel.predict(X_val, results_path)
            if config['common_parameters']['get_results']:
                gbm = GradientBoostingRegressorModel.record_scores(X_val, y_val, config['regression']['performance_metrics'], results_path)

        if config['regression']['regression_models']['xgBoost']:
            if config['common_parameters']['train']:
                xgb = XGBoostRegressionModel(X_train, y_train, X_val, y_val, config['regression']['regression_models']['xgb_params'], results_path)
            if config['common_parameters']['predict_new']:
                xgb = XGBoostRegressionModel.predict(X_val, results_path)
            if config['common_parameters']['get_results']:
                xgb = XGBoostRegressionModel.record_scores(X_val, y_val, config['regression']['performance_metrics'], results_path)

    elif config['common_parameters']['problem_type'] == 'classification':

        if config['classification']['classification_models']['logistic_regression']:
            if config['common_parameters']['train']:
                lr = LogisticRegressionModel(X_train, y_train, config['common_parameters']['n_runs'], results_path)
            if config['common_parameters']['predict_new']:
                lr = LogisticRegressionModel.predict(X_val, results_path)
            if config['common_parameters']['get_results']:
                lr = LogisticRegressionModel.record_scores(X_val, y_val, config['common_parameters']['n_runs'], config['classification']['performance_metrics'], results_path)

        if config['classification']['classification_models']['extra_trees']:
            if config['common_parameters']['train']:
                ext = ExtraTreesClassificationModel(X_train, y_train, X_val, y_val, config['common_parameters']['n_runs'], config['classification']['classification_models']['ext_params'], results_path)
            if config['common_parameters']['predict_new']:
                ext = ExtraTreesClassificationModel.predict(X_val, results_path)
            if config['common_parameters']['get_results']:
                ext = ExtraTreesClassificationModel.record_scores(X_val, y_val, config['common_parameters']['n_runs'], config['classification']['performance_metrics'], results_path)

        if config['classification']['classification_models']['random_forest']:
            if config['common_parameters']['train']:
                rf = RandomForestClassificationModel(X_train, y_train, X_val, y_val, config['common_parameters']['n_runs'], config['classification']['classification_models']['rf_params'], results_path)
            if config['common_parameters']['predict_new']:
                rf = RandomForestClassificationModel.predict(X_val, results_path)
            if config['common_parameters']['get_results']:
                rf = RandomForestClassificationModel.record_scores(X_val, y_val, config['common_parameters']['n_runs'], config['classification']['performance_metrics'], feature_names, results_path)
    else:
        raise Exception("Incorrect Problem Type Entered")
