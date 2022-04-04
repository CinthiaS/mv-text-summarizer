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

import matplotlib.pyplot as plt
import random

from multiprocessing import Process, Queue
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from src import pipeline_extract_features as pef
from src import utils

import joblib
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

def extract_features(sections, initial_batch, embed_dim, batch_len, path_base, name_file, verbose=False, dataset='arxiv'):
    
    
    features_columns = ['sentences', 'count_one_gram', 'count_two_gram', 'count_three_gram', 
                        'tfisf', 'position_score', 'sentence_len', 'count_postag',
                        'count_ner', 'text_rank', 'lex_rank', 'lsa_rank', 'dist_centroid', 'articles']
    
    scores_columns = ['rouge_1', 'rouge_2', 'rouge_l', 'label', 'articles']
    
    embeddings_columns = [i for i in range(embed_dim)]
    embeddings_columns.append("article")
    
    print(name_file)

    vfunc = np.vectorize(pef.extract_features_file)
    
    if (dataset == 'plosonev2') or (dataset == 'arxiv') or (dataset == 'pubmed'):
        with open('{}/{}'.format(path_base, name_file, 'r')) as f:
            files = f.readlines()
    elif dataset == 'plosonev1':
        files = load_files(path_base, name_file)
    
    features_results = {'introduction': [], 'materials': [], 'conclusion': []}
    embeddings_results = {'introduction': [], 'materials': [], 'conclusion': []}

    count = 1
    
    with open(f'../../logs/log_{dataset}.txt','r') as f:
        processed_files = f.readlines()
        
    processed_files = [i.replace('\n', '') for i in processed_files]
    
    for file in files:
        
        data = json.loads(file)
        code = data.get('id')
        
        if not code in processed_files:
             
            with open(f'../../logs/log_{dataset}.txt','a') as f:
                f.write(code + '\n')

            for name_section in sections:
    
                features, embeddings = pef.extract_features_batches(
                    vfunc, data, name_file, name_section=name_section, verbose=verbose)
                
                if not features.empty:
                    features_results[name_section].append(features)
                    embeddings_results[name_section].append(embeddings)
                    count +=1
                
                if (count % 100 == 0) and (count != 0):
                    print(count)
              
                    for name_sections_save in sections:
                
                        utils.save_results(
                            features_results[name_sections_save], embeddings_results[name_sections_save],
                            batch=name_file.replace('.txt', f'_{count}'), name_section=name_sections_save, dataset=dataset, verbose=False)
                    
                    features_results = {'introduction': [], 'materials': [], 'conclusion': []}
                    embeddings_results = {'introduction': [], 'materials': [], 'conclusion': []}


                    count +=1
        else:
            count +=1
            
            if (count % 100 == 0) and (count != 0):
                print('pass')
            
        

def main(sections, path_base, path_pp_data, path_to_remove=[], dataset='plosone'):
    
    initial_batch=0
    
    if not os.path.exists(f'../../result_{dataset}'):
        os.makedirs(f'../../result_{dataset}')
        os.makedirs(f'../../result_{dataset}/introduction')
        os.makedirs(f'../../result_{dataset}/materials')
        os.makedirs(f'../../result_{dataset}/conclusion')
        
    start_time = datetime.now()
    
    name_files = os.listdir(path_pp_data)
    name_files.remove(".ipynb_checkpoints")
    
    for i in path_to_remove:
        name_files.remove(i)

    if (dataset == 'plosonev2') or (dataset == 'arxiv') or (dataset == 'pubmed'):
        Parallel(n_jobs=10)(delayed(extract_features)(sections, initial_batch, 300, 700, path_base, name_file, False, dataset) for name_file in name_files)
        
        #for name_file in name_files:
        #    extract_features(sections, initial_batch, 300, 700, path_base, name_file, False, dataset)
        
        
            
    elif dataset == 'plosonev1':
        batches = utils.create_batches(path, tam=1000)
        Parallel(n_jobs=12)(delayed(extract_features)(
            sections, initial_batch, embed_dim=300, batch_len=700, path_base=path_base, name_file=name_file,
            verbose=False, dataset=dataset) for batch in batches)

    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    