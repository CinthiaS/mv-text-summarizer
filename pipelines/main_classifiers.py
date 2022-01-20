import sys
sys.path.insert(1, '/scratch/cinthiasouza/mv-text-summarizer')

import pandas as pd
import numpy as np
import json
import pickle
import os
import joblib
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

path_base = "/scratch/cinthiasouza/mv-text-summarizer"
path_to_read="/scratch/cinthiasouza/mv-text-summarizer/result/{}/{}_*.csv"

from src import classifiers

def create_directories(path):
    
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs("{}/summaries".format(path))
        os.makedirs("{}/predictions".format(path))
        
def main(dataset, parameters, sections, name_models, num_test):
    
    for name_test, approach in tests.items():
    
        if approach == 'sv':
            index_Xtrain='X_train'
        elif approach == 'mv-f1':
            index_Xtrain='X_train_f1'
        elif approach == 'mv-f2':
            index_Xtrain= 'X_train_f2'
        elif approach == 'embed':
            index_Xtrain= 'X_train_embed'
        elif approach == 'mv-f3':
            index_Xtrain= 'X_train_f3'
        
        index_ytrain= 'y_train'

        path_to_save = "/scratch/cinthiasouza/mv-text-summarizer/notebook/{}".format(name_test)
        create_directories(path_to_save)
        
        if not os.path.exists("{}/test_{}".format(path_to_save, num_test)):
            os.makedirs("{}/test_{}".format(path_to_save, num_test))

        for name_model in name_models: 

            classifiers.pipeline_classifiers(
                dataset, parameters, sections, [name_model],
                index_Xtrain, index_ytrain, num_test, path_to_save)
            

if __name__ == '__main__':
    
    n_test=1
    
    with open('dataset/dataset_{}.pkl'.format('features'), 'rb') as fp:
        dataset = pickle.load(fp)
        
    with open('mv_tunning/parameters.pkl', 'rb') as fp:
        parameters = pickle.load(fp)
        
    tests = {'mv_models_f3':'mv-f3'}
    
    sections=['introduction', 'materials', 'conclusion']
    name_models = ['knn', 'rf', 'ab', 'gb', 'cb',  'mlp']
    
    Parallel(n_jobs=10)(delayed(main)(
        dataset, parameters, sections, name_models, num_test) for num_test in range(1, n_test+1))
 