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

    with open('batches.json', 'w') as f:
        json.dump(batches, f)

def create_batches(path_base, tam=45):

    files = os.listdir(path_base)
    batch_files = np.array_split(files,tam)

    return batch_files

def load_batches(files, path_base):

    disct_list = {}
    cont = 1

    section_1 = []
    section_2 = []
    section_3 = []
    section_4 = []
    keywords = []

    texts = loader.load_files(path_base, files)

    for i in texts:

        #name = files[cont-1].replace(".json", "")
        #disct_list[name] = i

        #if cont != save_each == 0:
        #    with open('data/{}.json'.format(i), 'w') as f:
        #        json.dump(disct_list, f)
        #    disct_list = {}
        #    cont = 1

        section_1.append(preprocess.format_intro(i.get('sec_abstract')))
        section_2.append(preprocess.format_intro(i.get('sec_introduction')))
        section_3.append(preprocess.format_intro(i.get('sec_materials_and_methods')))
        section_4.append(preprocess.format_intro(i.get('sec_results_and_conclusion')))
        keywords.append(i.get('sec_keyword'))

    return section_1, section_2, section_3, section_4, keywords

def save_results(all_features, all_scores, number_text, batch, name_section='intro', verbose=False):
    
    features_df = pd.concat(all_features)
    scores_df = pd.concat(all_scores)

    features_df.to_csv("../result/{}/features_{}_batch_{}.csv".format(
       name_section, number_text, batch), index=False)
    scores_df.to_csv("../result/{}/scores_{}_batch_{}.csv".format(
       name_section, number_text, batch), index=False)