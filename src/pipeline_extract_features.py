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
    vfunc, batch, path_base, name_section, features_columns, scores_columns, embeddings_columns, verbose=True):
    

    my_file = Path("../result/{}/features_{}.csv".format(name_section, batch))
    
    if not my_file.is_file():
        
        #Load Files
        df = utils.load_batches(
            batch, path_base, name_section)
        
        if verbose:
            print("Total de arquivos: {} \n".format(df.shape))

        
        try:
            #print(batch)
            features, scores, embeddings = vfunc(df['abstract'], df['texts'], df['keywords'], df['name_files'])
        
        # Convert numpy array to dataframe
            features = [array_to_df(features[i], features_columns) for i in range(len(features))]
            scores = [array_to_df(scores[i], scores_columns) for i in range(len(scores))]
            embeddings = [array_to_df(embeddings[i], embeddings_columns) for i in range(len(scores))]
        
            if len(features[0]) > 0:            
                utils.save_results(
                    features, scores, embeddings, batch=batch[0].replace('.json', ''),
                    name_section=name_section, verbose=False)
        except KeyError as error:
            pass

    
def extract_features_file(reference, section, keywords, number_text, verbose=False):
    
    
    xml = preprocess.format_xml(str(section))
    text = preprocess.format_text(str(section), post_processing=False)
    reference = preprocess.format_text(str(reference), post_processing=True)


    bibs = extract_features.get_citations(xml)
    text = preprocess.replace_bib(text, bibs)
    text = preprocess.format_text(text, post_processing=True)

    soup = BeautifulSoup(text)
    text = soup.get_text()

    sentences = tokenizer.split_sentences([text])
    sentences = list(map(str, sentences[0]))
    sentences = preprocess.format_sentences(sentences)

    try:

        features, embeddings = create_features_df.main(sentences, xml, keywords, nlp_sm, nlp_md)
        features_df = create_features_df.format_df (sentences, features)
        features_df['number_text'] = [number_text]*len(features_df)
        embeddings['numbert_tex'] = [number_text]*len(features_df)

        sentences_ref = tokenizer.split_sentences([reference])
        sentences_ref = list(map(str, sentences_ref[0]))

        scores_df, label = transform_data.main_create_label(sentences, sentences_ref, rouge)
        scores_df['label'] = label
        scores_df['number_text'] = [number_text]*len(scores_df)
    
        features = features_df.to_numpy(dtype=object)
        scores = scores_df.to_numpy(dtype=object)
        embeddings = embeddings.to_numpy(dtype=object)

        return features, scores, embeddings

    except IndexError as error:
        out = pd.DataFrame().to_numpy(dtype=object)
        return out, out, out
    except ValueError as error:
        out = pd.DataFrame().to_numpy(dtype=object)
        return out, out, out
    except KeyError as error:
        out = pd.DataFrame().to_numpy(dtype=object)
        return out, out, out

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
    