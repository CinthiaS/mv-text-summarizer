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

def read_features(path, name="features"):

    print(path)
    path_files = glob.glob(path)
    features = []
    scores = []
    
    for p in path_files:
        try:
            features.append(pd.read_csv(p))
            scores.append(pd.read_csv(p.replace(name, "scores")))
        except:
            pass
        
              
    features = pd.concat(features).reset_index(drop=True)
    scores = pd.concat(scores).reset_index(drop=True)
    
    features.to_csv("all_features.csv", index=False)
    scores.to_csv("all_scores.csv", index=False)
    
    return features, scores



