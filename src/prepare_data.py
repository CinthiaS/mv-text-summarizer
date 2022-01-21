import pandas as pd 
import numpy as np
import json

from joblib import Parallel, delayed
from imblearn.under_sampling import RandomUnderSampler

from src import normalization
from src import classifiers
from src import utils_classification as utils_clf
from src import utils
from src import loader
from src import evaluate_classifiers as ev

        
def prepare_concat_data(features, scores, columns_name, label_column='bin'):
    
    summ_items = list(pd.read_csv("dataset/indices_summ.csv")['summ'])
    
    #data = utils.join_dataset(features, scores)
    train, test = utils.split_dataset (data, summ_items)
    
    train = utils_clf.transform_to_classification(train)
    test = utils_clf.transform_to_classification(test)
    
    train.to_csv("train.csv")
    test.to_csv("test.csv")
    
    X_train, y_train = RandomUnderSampler(random_state=47).fit_resample(train[columns_name], train[label_column])
    
    X_test = test[columns_name]
    y_test = test[label_column]
    
    
    return X_train, X_test, y_train, y_test

def concat_sections(section, train_columns, under_columns, summ_items, path_to_read, name_csv="features", label_column='bin'):
    
    dataset_intro = pd.read_csv('dataset/dataset_{}.csv'.format("introduction"))
    dataset_mat = pd.read_csv('dataset/dataset_{}.csv'.format("materials"))
    dataset_conc = pd.read_csv('dataset/dataset_{}.csv'.format("conclusion"))
    
    dataset = pd.concat([dataset_intro, dataset_mat, dataset_conc])
    
    train, test = utils.split_dataset (dataset, summ_items)
    
    train = utils_clf.transform_to_classification(train)
    test = utils_clf.transform_to_classification(test)
    
    X_train, y_train = RandomUnderSampler(random_state=47).fit_resample(train[under_columns], train[label_column])
    
    train = pd.DataFrame(X_train, columns=under_columns)
    train[label_column] = y_train
    
    X_train = X_train[train_columns]
    
    X_train, scaler = normalization.scale_fit_transform(X_train, section='scaler_{}_class'.format(section))
    X_train, y_train = utils_clf.shuffle_dataset(X_train, y_train)
    
    X_test = scaler.transform(test[train_columns])
    y_test = test[label_column]
    
    return X_train, X_test, y_train, y_test, train, test

def create_data_classification(
    section, train_columns, under_columns, summ_items, path_to_read, name_csv="features", label_column='bin'):
    
    dataset = pd.read_csv('dataset/dataset_{}.csv'.format(section))
    
    train, test = utils.split_dataset (dataset, summ_items)
    
    train = utils_clf.transform_to_classification(train)
    test = utils_clf.transform_to_classification(test)
    
    X_train, y_train = RandomUnderSampler(random_state=47).fit_resample(train[under_columns], train[label_column])
    
    train = pd.DataFrame(X_train, columns=under_columns)
    train[label_column] = y_train
    
    X_train = X_train[train_columns]
    
    X_train, scaler = normalization.scale_fit_transform(X_train, section='scaler_{}_class'.format(section))
    X_train, y_train = utils_clf.shuffle_dataset(X_train, y_train)
    
    X_test = scaler.transform(test[train_columns])
    y_test = test[label_column]
    
    return X_train, X_test, y_train, y_test, train, test

def main_create_dataset(train_columns, under_columns, sections, path_to_read, name_csv):
    
    summ_items = list(pd.read_csv("dataset/indices_summ.csv")['summ'])
    print(len(summ_items))

    dataset = {}

    for section in sections:

        if section == 'concat':
                X_train, X_test, y_train, y_test, train, test = concat_sections(
                section=section, train_columns=train_columns, under_columns=under_columns, summ_items=summ_items, path_to_read=path_to_read,
                name_csv=name_csv, label_column='bin')
            
        else:
            X_train, X_test, y_train, y_test, train, test = create_data_classification(
                section=section, train_columns=train_columns, under_columns=under_columns, summ_items=summ_items,
                path_to_read=path_to_read, name_csv=name_csv, label_column='bin')
        
        
        dataset[section] = {"X_train_features": X_train,
                            "X_test_features": X_test,
                            "y_train": y_train,
                            "y_test": y_test,
                            "X_train_nf": train,
                            "X_test_nf": test}

        
    return dataset

