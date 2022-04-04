import sys
sys.path.insert(1, '/scratch/cinthiasouza/mv-text-summarizer')

import itertools
import re

import glob, os
import pandas as pd
import json
import spacy
import nltk
import numpy as np
import json
import seaborn as sns
import pickle

from datetime import datetime

from bs4 import BeautifulSoup
from pysbd.utils import PySBDFactory
import math

from sumeval.metrics.rouge import RougeCalculator
rouge = RougeCalculator(stopwords=True, lang="en")
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings("ignore")

from src import utils
from src import prepare_data


def main(
    train_columns, under_columns, sections, summ_items, path_to_read, path_to_save,
    name_csv, format_dataset=True, verbose=True):
    
    start_time = datetime.now()
    
    if format_dataset:
        if verbose:
            print("Preparando dataset para os classificadores")
        dataset = prepare_data.main_create_dataset(
            train_columns, under_columns, sections, summ_items, path_to_read, name_csv)
    else:
        if verbose:
            print("Carregando dataset")
        dataset = utils.load_json(name='dataset', path=path)
    
    if verbose:
        print("Treinamento dos modelos")
        
    with open('{}/dataset_{}.pkl'.format(path_to_save, name_csv), 'wb') as fp:
        pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
        
    return dataset
