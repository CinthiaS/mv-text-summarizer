import json
import os
import numpy as np
import pandas as pd

from src import preprocess
from src import loader


def check(i):
    if i == None:
        return 0
    else:
        return i


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



def load_batches(files, path_base, name_section):

    disct_list = {}
    cont = 1

    section_1 = []
    text = []
    keywords = []

    texts, files = loader.load_files(path_base, files)

    for i in texts:
        
        if name_section == 'introduction':
            section_1.append(preprocess.format_intro(i.get('sec_abstract')))
            text.append(preprocess.format_intro(i.get('sec_introduction')))
        elif name_section == 'materials':
            section_1.append(preprocess.format_intro(i.get('sec_abstract')))
            text.append(preprocess.format_intro(i.get('sec_materials_and_methods')))
        elif name_section == 'conclusion':
            section_1.append(preprocess.format_intro(i.get('sec_abstract')))
            text.append(preprocess.format_intro(i.get('sec_results_and_conclusion')))

        keywords.append(i.get('sec_keyword'))
        
    df = pd.DataFrame({'abstract': section_1, 'texts': text, 'keywords': keywords, 'name_files': files })

    return df

def save_results(all_features, all_scores, all_embeddings, batch, name_section='intro', verbose=False):
    
    try:
    
        features_df = pd.concat(all_features)
        scores_df = pd.concat(all_scores)
        #embeddings_df = pd.concat(all_embeddings)

        features_df.to_csv("../result/{}/features_{}.csv".format(
           name_section, batch), index=False)
        scores_df.to_csv("../result/{}/scores_{}.csv".format(
           name_section, batch), index=False)
        #embeddings_df.to_csv("../result/{}/embeddings_{}.csv".format(
        #   name_section, batch), index=False)
    
    except ValueError:
        
        pass

def join_dataset(X, y):
    
    features = X.copy()
    features['rouge_1'] = y['rouge_1']
    
    return features

def split_dataset (features, summ_items):

    train = features.loc[~features['articles'].isin(summ_items)]
    train.reset_index(inplace=True, drop=True)

    test = features.loc[features['articles'].isin(summ_items)]
    test.reset_index(inplace=True, drop=True)
    
    return train, test


def save_json(dataset, name, path='.'):

    with open('{}/{}.json'.format(path, name), 'w') as fp:
        json.dump(dataset, fp)
        
def load_json(name, path='.'):

    dataset = json.loads(open ('{}/{}.json'.format(path,name), "r").read())
    return dataset
    
def save_results_eval(results, path='.'):
    
    for i in results.keys():
        
        df = pd.DataFrame(results[i])
        df.to_csv("{}/eval_{}.csv".format(path, i), index=False)
