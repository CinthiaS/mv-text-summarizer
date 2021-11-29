import pandas as pd 
import numpy as np
import json

from joblib import Parallel, delayed
from imblearn.under_sampling import RandomUnderSampler

from src import normalization
from src import ensemble_tree_models as classifiers
from src import utils_classification as utils_clf
from src import utils
from src import loader
from src import evaluate_classifiers as ev

        
def prepare_concat_data(features, scores, columns_name, label_column='bin'):
    
    summ_items = list(pd.read_csv("indices_summ.csv")['summ'])
    
    data = utils.join_dataset(features, scores)
    train, test = utils.split_dataset (data, summ_items)
    
    train = utils_clf.transform_to_classification(train)
    test = utils_clf.transform_to_classification(test)
    
    train.to_csv("train.csv")
    test.to_csv("test.csv")
    
    X_train, y_train = RandomUnderSampler().fit_resample(train[columns_name], train[label_column])
    
    X_test = test[columns_name]
    y_test = test[label_column]
    
    
    return X_train, X_test, y_train, y_test

def concat_sections(dataset, columns_name, path_to_read, name_csv="features", label_column='bin'):
    
    features_intro, scores_intro = loader.read_features(
        path=path_to_read.format('introduction', name_csv), name=name_csv)
    features_mat, scores_mat = loader.read_features(
        path=path_to_read.format('materials', name_csv), name=name_csv)
    features_conc, scores_conc = loader.read_features(
        path=path_to_read.format('conclusion', name_csv), name=name_csv)
    
    X_train_intro, X_test_intro, y_train_intro, y_test_intro = prepare_concat_data(features_intro, scores_intro, columns_name)
    X_train_mat, X_test_mat, y_train_mat, y_test_mat = prepare_concat_data(features_mat, scores_mat, columns_name)
    X_train_conc, X_test_conc, y_train_conc, y_test_conc = prepare_concat_data(features_conc, scores_conc, columns_name)
    
    X_train = pd.concat([X_train_intro, X_train_mat, X_train_conc])
    X_test = pd.concat([X_test_intro, X_test_mat, X_test_conc])
    y_train = pd.concat([y_train_intro, y_train_mat, y_train_conc])
    y_test = pd.concat([y_test_intro, y_test_mat, y_test_conc])
    
    X_train, scaler = normalization.scale_fit_transform(X_train, section='scaler_{}_class'.format(dataset))
    X_train, y_train = utils_clf.shuffle_dataset(X_train, y_train)
    
    X_test = scaler.transform(X_test[columns_name])
    
    return X_train, X_test, y_train, y_test

def create_data_classification(dataset, columns_name, summ_items, path_to_read, name_csv="features", label_column='bin'):
    
    features, scores = loader.read_features(path=path_to_read.format(dataset, name_csv),name=name_csv)
    
    data = utils.join_dataset(features, scores)
    train, test = utils.split_dataset (data, summ_items)
    
    train = utils_clf.transform_to_classification(train)
    test = utils_clf.transform_to_classification(test)
    
    X_train, y_train = RandomUnderSampler().fit_resample(train[columns_name], train[label_column])
    
    X_train, scaler = normalization.scale_fit_transform(X_train, section='scaler_{}_class'.format(dataset))
    X_train, y_train = utils_clf.shuffle_dataset(X_train, y_train)
    
    X_test = scaler.transform(test[columns_name])
    y_test = test[label_column]
    
    return X_train, X_test, y_train, y_test, train, test

def main_create_dataset(columns_name, sections, path_to_read, name_csv):
    
    summ_items = list(pd.read_csv("indices_summ.csv")['summ'])

    dataset = {}

    for section in sections:

        if section == 'concat':
            X_train, X_test, y_train, y_test = concat_sections(
                dataset=section, columns_name=columns_name, path_to_read=path_to_read,
                name_csv=name_csv, label_column='bin')
            
        else:
            X_train, X_test, y_train, y_test, train, test = create_data_classification(
                dataset=section, columns_name=columns_name, summ_items=summ_items,
                path_to_read=path_to_read, name_csv=name_csv, label_column='bin')
        
        dataset[section] = [X_train, X_test, y_train, y_test, train, test]

        
    return dataset

