import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def data_loader(data_path, header):
    header = 0 if header else None
    data = pd.read_csv(data_path, header=header)
    return data


def split_X_y(data, target_variable):
    X = data.loc[:, data.columns != target_variable]
    y = data.loc[:, data.columns == target_variable]
    return X,y


def convert_categorical(X, y):
    cols = X.columns
    num_cols = X._get_numeric_data().columns
    cols_to_transform = list(set(cols) - set(num_cols))
    encoded_X = pd.get_dummies(X, columns = cols_to_transform)

    if y.dtypes[0] != 'int' and y.dtypes[0] != 'float':
        le = LabelEncoder()
        encoded_y = pd.DataFrame()
        encoded_y[y.columns[0]] = le.fit_transform(y.iloc[:,0])

    return encoded_X, encoded_y


def train_val_split(features, labels, split_fraction=0.75):
    X_train, X_val, y_train, y_val = train_test_split(features, labels, train_size=float(split_fraction))
    return np.array(X_train), np.array(X_val), np.squeeze(np.array(y_train)), np.squeeze(np.array(y_val))


def process_data(data_path, config, args):
    data = data_loader(data_path, config['common_parameters']['has_header'])
    X, y = split_X_y(data, config['common_parameters']['target_variable'])
    X, y = convert_categorical(X, y)
    transformed_data = pd.concat([X,y], axis=1)
    dataset_numeric_path = args.output_path + '/data/dataset-numeric.csv'
    transformed_data.to_csv(dataset_numeric_path, index=False)
    X_train, X_val, y_train, y_val = train_val_split(X, y, config['common_parameters']['train_val_split'])
    
