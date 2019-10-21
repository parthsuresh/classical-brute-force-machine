import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def data_loader(data_path, header, columns):
    header = 0 if header else None
    data = pd.read_csv(data_path, header=header, usecols=columns)
    return data


def impute_data(X_train, X_val, impute_method, variables, categorical_variables, target_variable):
    if len(variables) <= 1: raise Exception("Incorrect number of variables in Config File")
    if len(categorical_variables) == 1: categorical_variables = [categorical_variables]
    numerical_variables = list(set(variables) - set(categorical_variables) - set([target_variable]))

    if len(numerical_variables) >= 1:
        numeric_X_train = X_train[numerical_variables]
        numeric_X_val = X_val[numerical_variables]
        if impute_method == 'mean':
            imp = SimpleImputer(strategy='mean')
        elif impute_method == 'median':
            imp = SimpleImputer(strategy='median')
        elif impute_method == 'most_frequent':
            imp = SimpleImputer(strategy='most_frequent')
        else:
            raise Exception("Invalid Imputation Method in Config File")

        imp.fit(numeric_X_train)
        numeric_X_train_imputed = pd.DataFrame(imp.transform(numeric_X_train), columns=numeric_X_train.columns)
        numeric_X_val_imputed = pd.DataFrame(imp.transform(numeric_X_val), columns = numeric_X_val.columns)

    if len(categorical_variables) >= 1:
        categorical_X_train = X_train[categorical_variables]
        categorical_X_val = X_val[categorical_variables]
        cat_imp = SimpleImputer(strategy='most_frequent')
        cat_imp.fit(categorical_X_train)
        categorical_X_train_imputed = pd.DataFrame(cat_imp.transform(categorical_X_train), columns=categorical_X_train.columns)
        categorical_X_val_imputed = pd.DataFrame(cat_imp.transform(categorical_X_val), columns=categorical_X_val.columns)

    if len(categorical_variables) >= 1 and len(numerical_variables) >= 1:
        X_train_imputed = pd.concat(numeric_X_train_imputed, categorical_X_train_imputed)
        X_val_imputed = pd.concat(numeric_X_val_imputed, categorical_X_val_imputed)
    elif len(categorical_variables) >= 1:
        X_train_imputed = categorical_X_train_imputed
        X_val_imputed = categorical_X_val_imputed
    else:
        X_train_imputed = numeric_X_train_imputed
        X_val_imputed = numeric_X_val_imputed

    return X_train_imputed, X_val_imputed

def split_X_y(data, target_variable):
    X = data.loc[:, data.columns != target_variable]
    y = data.loc[:, data.columns == target_variable]
    return X,y


def convert_categorical(X, y, problem_type, cols_to_transform=[]):
    if cols_to_transform:
        encoded_X = pd.get_dummies(X, columns = cols_to_transform)
    else:
        encoded_X = pd.DataFrame(X)

    if problem_type == "classification":
        le = LabelEncoder()
        encoded_y = pd.DataFrame()
        encoded_y[y.columns[0]] = le.fit_transform(y.iloc[:,0])
    else:
        encoded_y = y

    return encoded_X, encoded_y


def train_val_split(features, labels, split_fraction=0.75):
    X_train, X_val, y_train, y_val = train_test_split(features, labels, train_size=float(split_fraction))
    return X_train, X_val, y_train, y_val


def process_data(data_path, config, args):
    print('Processing Data...')
    data = data_loader(data_path, config['common_parameters']['has_header'], config['common_parameters']['variables'])

    X, y = split_X_y(data, config['common_parameters']['target_variable'])

    X_train, X_val, y_train, y_val = train_val_split(X, y, config['common_parameters']['train_val_split'])

    X_train, X_val = impute_data(X_train, X_val,
                    config['common_parameters']['impute_method'],
                    config['common_parameters']['variables'],
                    config['common_parameters']['categorical_variables'],
                    config['common_parameters']['target_variable'])

    X_train, y_train = convert_categorical(X_train, y_train, config['common_parameters']['problem_type'], config['common_parameters']['categorical_variables'])
    X_val, y_val = convert_categorical(X_val, y_val, config['common_parameters']['problem_type'], config['common_parameters']['categorical_variables'])

    transformed_training_data = pd.concat([X_train,y_train], axis=1)
    training_dataset_numeric_path = args.output_path + '/results/data/training-dataset-numeric.csv'
    transformed_training_data.to_csv(training_dataset_numeric_path, index=False)

    transformed_validation_data = pd.concat([X_val,y_val], axis=1)
    validation_dataset_numeric_path = args.output_path + '/results/data/validation-dataset-numeric.csv'
    transformed_validation_data.to_csv(validation_dataset_numeric_path, index=False)

    print('Processing Completed.')
    return np.array(X_train), np.array(X_val), np.squeeze(np.array(y_train)), np.squeeze(np.array(y_val))
