import os
import json
import pandas as pd
import glob

from src import preprocess

def load_files(path_base, files):

    texts = []
    for file in files:
        texts.append(json.load(open('{}/{}'.format(path_base, file))))

    return texts, files

def get_section(texts, section_name, preprocessed=True):

    if preprocessed:
        section = [preprocess.format_intro(i.get(section_name)) for i in texts]
    else:
        section = [i.get(section_name) for i in texts]

    return section

def read_features(path="../result/features_*.csv"):

    path_files = glob.glob(path)
    features = []
    scores = []
    
    for p in path_files:
        try:
            features.append(pd.read_csv(p))
            scores.append(pd.read_csv(p.replace("features", "scores")))
        except:
            pass
              
    features = pd.concat(features).reset_index(drop=True)
    scores = pd.concat(scores).reset_index(drop=True)
    
    return features, scores



