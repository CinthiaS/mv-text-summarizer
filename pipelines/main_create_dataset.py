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

def main(path_to_read, name_csv="features", format_dataset=True, verbose=True):
    
    train_columns = ['text_rank', 'lex_rank', 'count_one_gram', 'count_two_gram',
       'count_three_gram', 'count_article_keywords',
       'tf-isf', 'position_score', 'paragraph_score',
       'length_score', 'pos_score', 'ner_score', 'dist_centroid']
    
    under_columns = ['sentences', 'articles', 'text_rank', 'lex_rank', 'count_one_gram', 'count_two_gram',
       'count_three_gram', 'count_article_keywords',
       'tf-isf', 'position_score', 'paragraph_score',
       'length_score', 'pos_score', 'ner_score', 'dist_centroid', 'rouge_1', 'bin']

    sections=['introduction', 'materials', 'conclusion', 'concat']

    if format_dataset:
        if verbose:
            print("Preparando dataset para os classificadores")
        dataset = prepare_data.main_create_dataset(
            train_columns, under_columns, sections, path_to_read, name_csv)
    else:
        if verbose:
            print("Carregando dataset")
        dataset = utils.load_json(name='dataset', path=path)
    
    if verbose:
        print("Treinamento dos modelos")
        
    with open('dataset_{}.pkl'.format(name_csv), 'wb') as fp:
        pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
    return dataset

if __name__ == '__main__':
    
    path_base = "mv-text-summarizer"
    path_to_read="mv-text-summarizer/result/{}/{}_*.csv"
    path_to_read="mv-text-summarizer/dataset/"
    
    start_time = datetime.now()
    
    dataset  = main(verbose=True, path_to_read=path_to_read, name_csv='features')
    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))