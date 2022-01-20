import sys
sys.path.insert(1, '/scratch/cinthiasouza/mv-text-summarizer')

import itertools
import re

import glob, os
import pandas as pd
import json
#import spacy
import nltk
import numpy as np
import json
import seaborn as sns
import pickle
import math

from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier 

from src import tunning_hyperparametrs as th

import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Process, Queue

path_base = "/scratch/cinthiasouza/mv-text-summarizer"
path_to_read="/scratch/cinthiasouza/mv-text-summarizer/result/{}/{}_*.csv"

parameters_tunning = {
    
 'rf': {'n_estimators': [10, 25, 50, 100],
              'min_samples_leaf':[2500, 5000, 1000],
              'min_samples_split':[5000, 10000, 2000],
              'max_depth':[2, 3, 5, 10]
              },
 'gb': {'n_estimators': [10, 25, 50, 100],
              'min_samples_leaf':[2500, 5000, 1000],
              'min_samples_split':[5000, 10000, 2000],
              'max_depth':[2, 3, 5, 10]
              },
'cb': {'iterations': [10, 25, 50, 100],
            'learning_rate': [0.01],
            'depth': [2, 3, 5, 10],
            'min_data_in_leaf':[2500, 5000, 1000]},
 'knn':  {'n_neighbors': [3, 5, 10]},
    
 'ab': {'n_estimators': [10, 25, 50, 100]}}


if __name__ == '__main__':

    X_train=''
    X_test=""
    

    l1 = Queue()
    p1 = Process(
        target=th.pipeline_tunning, args=(
        dataset, {'knn': KNeighborsClassifier()},
        sections, parameters_tunning, path_to_save, 0, 2))
    
    l2 = Queue()
    p2 = Process(
        target=th.pipeline_tunning, args=(
        dataset, {'rf': RandomForestClassifier()},
        sections, parameters_tunning, path_to_save, 0, 2))
    
    l3 = Queue()
    p3 = Process(
        target=th.pipeline_tunning, args=(
        dataset, {'ab': AdaBoostClassifier()},
        sections, parameters_tunning, path_to_save, 0, 2))
    
    l4 = Queue()
    p4 = Process(
        target=th.pipeline_tunning, args=(
        dataset, {'cb': CatBoostClassifier()},
        sections, parameters_tunning, path_to_save, 0, 2))
    
    l5 = Queue()
    p5 = Process(
        target=th.pipeline_tunning, args=(
        dataset, {'gb': GradientBoostingClassifier()},
        sections, parameters_tunning, path_to_save, 0, 2))
    

    p1.start()
    p2.start()  
    p3.start()  
    p4.start()  
    p5.start()  