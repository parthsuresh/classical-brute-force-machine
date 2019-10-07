import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def data_loader(data_path, header=None):
    header = 0 if header else None
    data = pd.read_csv(data_path, header=header)
    return data

def split_feature_labels(data, target_variable):
    features = data.loc[:, data.columns != target_variable]
    labels = data.loc[:, target_variable]
    return features,labels

def train_val_split(features, labels, split_fraction=0.75):
    return train_test_split(features, labels, test_size=1-split_fraction)

def process_data(data_path, config):
    data = data_loader(data_path, config['common_parameters']['has_header'])
    features, labels = split_feature_labels(data, config['common_parameters']['target_variable'])
    X_train, X_val, y_train, y_val = train_test_split(features, labels, config['common_parameters']['train_val_split'])
