import os
import warnings

import argparse
import numpy as np

from config_parse import parse_config
from data.data_preprocessing import process_data
from models.linear_regression import LinearRegressionModel
from models.gbm_regression import GradientBoostingRegressorModel

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

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    if not os.path.exists(model_scores_path):
        os.mkdir(model_scores_path)
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)

    config = parse_config(args.config_path)
    X_train, X_val, y_train, y_val = process_data(args.data_path, config, args)
    f = open(model_scores_path+"/metric.txt", "a")
    f.close()

    # Models
    if config['common_parameters']['problem_type'] == 'regression':

        if config['regression']['regression_models']['linear_regression']:
            if config['common_parameters']['train']:
                lr = LinearRegressionModel(X_train, y_train, results_path)
            if config['common_parameters']['predict']:
                lr = LinearRegressionModel.predict(X_val, results_path)
            if config['common_parameters']['record']:
                score = LinearRegressionModel.record_scores(X_val, y_val, config['regression']['performance_metrics'], results_path)

        if config['regression']['regression_models']['gbm']:
            if config['common_parameters']['train']:
                gbm = GradientBoostingRegressorModel(X_train, y_train, X_val, y_val, config['regression']['regression_models']['gbm_params'], results_path)
            if config['common_parameters']['predict']:
                gbm = GradientBoostingRegressorModel.predict(X_val, results_path)
            if config['common_parameters']['record']:
                lr = GradientBoostingRegressorModel.record_scores(X_val, y_val, config['regression']['performance_metrics'], results_path)

        if config['regression']['regression_models']['xgBoost']:
            if config['common_parameters']['train']:
                gbm = XGBoostRegressionModel(X_train, y_train, X_val, y_val, config['regression']['regression_models']['xgb_params'], results_path)
            if config['common_parameters']['predict']:
                gbm = XGBoostRegressionModel.predict(X_val, results_path)
            if config['common_parameters']['record']:
                lr = XGBoostRegressionModel.record_scores(X_val, y_val, config['regression']['performance_metrics'], results_path)
