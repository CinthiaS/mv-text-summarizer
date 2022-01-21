import sys
sys.path.insert(1, '/scratch/cinthiasouza/mv-text-summarizer')

import glob, os
import pandas as pd
import json
import spacy
import nltk
import numpy as np
import json
import pickle
from datetime import datetime

from bs4 import BeautifulSoup
from pysbd.utils import PySBDFactory
import math

from sumeval.metrics.rouge import RougeCalculator
rouge = RougeCalculator(stopwords=True, lang="en")
import matplotlib.pyplot as plt
import random

from multiprocessing import Process, Queue
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from src import pipeline_extract_features as pef

import joblib
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

def extract_features(name_section, initial_batch, embed_dim, batch_len, path_base, verbose):
    
    features_columns = ['sentences', 'sentences_text', 'sentences_lex', 'sentences_lsa',
                        'count_one_gram', 'count_two_gram', 'count_three_gram', 
                        'count_article_keywords', 'tf-isf', 'position_score', 
                        'paragraph_score', 'number_citations', 'length_score', 'pos_score',
                        'ner_score', 'dist_centroid', 'articles']
    
    scores_columns = ['rouge_1', 'rouge_2', 'rouge_l', 'label', 'articles']
    
    embeddings_columns = [i for i in range(embed_dim)]
    embeddings_columns.append("article")
    
    batche_files = os.listdir(path_base)

    print("Name section: " + name_section)
    vfunc = np.vectorize(pef.extract_features_file)
    
    print("Iniciando a extração de features...")
    
    for batch in batche_files:
        
        print(batch)

    
        pef.extract_features_batches(
            vfunc, [batch], path_base, name_section=name_section, features_columns=features_columns,
            scores_columns=scores_columns, embeddings_columns=embeddings_columns, verbose=verbose)
 


def main(sections, path_base, path_pp_data):
    
    initial_batch=0
    
    if not os.path.exists('../result'):
        os.makedirs('../result')
        os.makedirs('../result/introduction')
        os.makedirs('../result/materials')
        os.makedirs('../result/conclusion')
        
    start_time = datetime.now()
        
    Parallel(n_jobs=5)(delayed(extract_features)(s, initial_batch, 300, 700, path[0], False) for s in sections)
    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    