import sys
sys.path.insert(1, 'src')

import glob, os
import pandas as pd
import json
import spacy
import nltk
import numpy as np
import json

import seaborn as sns
import pickle
from pathlib import Path


from bs4 import BeautifulSoup
from pysbd.utils import PySBDFactory
import math

from sumeval.metrics.rouge import RougeCalculator
rouge = RougeCalculator(stopwords=True, lang="en")

from timeit import default_timer as timer  

from src import preprocess
from src import extract_features
from src import tokenizer
from src import create_features_df
from src import transform_data
from src import loader
from src import utils
#from src import gradient_boost
#from src import random_forest
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

nlp_sm = spacy.load('en_core_web_sm')
nlp_md = spacy.load('en_core_web_md')

def array_to_df(array, columns_name):
    
    if (array.size != 0):
        return pd.DataFrame(array, columns=columns_name)

    return pd.DataFrame(columns=columns_name)

def extract_features_batches(
    vfunc, data, path_base, name_section, verbose=True):
    
    features, features_columns,  embeddings, embeddings_columns = extract_features_file(
            data.get('abstract'), data.get(name_section), data.get('keywords'), data.get('id'))
        
    
    features = array_to_df(features, features_columns)
    embeddings = array_to_df(embeddings, embeddings_columns) 
        
    return features, embeddings

def extract_features_file(reference, sentences, keywords, number_text, verbose=False):
    

    reference = preprocess.format_text(str(reference), post_processing=True)

    df = pd.DataFrame()
    df['sentences'] = sentences
    df['sentences'] = df['sentences'].apply(preprocess.clean_text)
    
    sentences = list(df['sentences'])

    if not df.empty and len(df) > 2:
    
        features_df, embeddings_df = create_features_df.main(df, nlp_sm, nlp_md)
    
        features_df['articles'] = [number_text]*len(features_df)
        embeddings_df['articles'] = [number_text]*len(embeddings_df)

        sentences_ref = tokenizer.split_sentences([reference])
        sentences_ref = list(map(str, sentences_ref[0]))

        features_df = transform_data.main_create_label(features_df, sentences_ref, rouge)
    
        features = features_df.to_numpy(dtype=object)
        embeddings = embeddings_df.to_numpy(dtype=object)

        return features, features_df.columns, embeddings, embeddings_df.columns
    
    else:
        out = pd.DataFrame().to_numpy(dtype=object)
        return out, [], out, []



def save_batches(batch_files):
    
    batches = {}
    for i in range(len(batch_files)):
        batches[i+1] =  list(batch_files[i])

    with open('batches_500.json', 'w') as f:
        json.dump(batches, f)

def create_batches(path_base, tam=45):

    files = os.listdir(path_base)
    batch_files = np.array_split(files,tam)

    return batch_files

"""
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_section', "-n",required=True)
    parser.add_argument('--initial_batch', "-b", default=0)
    parser.add_argument('--path_base', "-p", required=True)
    parser.add_argument('--embed_dim', "-e", default=300)
    parser.add_argument('--batch_len', "-bl", default=300)
    parser.add_argument('--verbose', "-v", default=False)
     
    args = parser.parse_args()"""