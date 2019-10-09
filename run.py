import os

import argparse

from config_parse import parse_config
from data.data_preprocessing import process_data
from models.linear_regression import LinearRegressionModel
from metrics.regression_metrics import rmse
from metrics.regression_metrics import mae
from metrics.regression_metrics import r2
from metrics.regression_metrics import pearson_correlation


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
    config = parse_config(args.config_path)
    X_train, X_val, y_train, y_val = process_data(args.data_path, config, args)

    if config['common_parameters']['problem_type'] == 'regression':
        if config['regression']['regression_models']['linear_regression']:
            lr = LinearRegressionModel(X_train, y_train)
            preds = lr.predict(X_val)

    # Metrics
    if config['regression']['performance_metrics']['rmse']:
        print('RMSE : ', rmse(preds, y_val))
    if config['regression']['performance_metrics']['mae']:
        print('MAE: ', mae(preds, y_val))
    if config['regression']['performance_metrics']['r_squared']:
        print('R2: ', r2(preds, y_val))
    #if config['regression']['performance_metrics']['pearson_correlation']:
    #    print('Pearson correlation : ', pearson_correlation(preds, y_val))
