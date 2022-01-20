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
import pickle

import warnings
warnings.filterwarnings("ignore")

from src import transform_data
from src import summarization
from src import normalization
from src import classifiers
from src import evaluate_classifiers as ev
from src import display_results as dr
from src import utils_classification as utils

import joblib
from joblib import Parallel, delayed
from src import pipeline_extract_features as pef


def main (
    dataset, tests, num_test, references_df, sections, name_models, columns):
    
    for name_test, approach in tests.items():

        if approach == 'sv':
            index_Xtest='X_test'
        elif approach == 'mv-f1':
            index_Xtest='X_test_f1'
        elif approach == 'mv-f2':
            index_Xtest= 'X_test_f2'
        elif approach == 'embed':
            index_Xtest= 'X_test_embed'

        path_to_save = "/scratch/cinthiasouza/mv-text-summarizer/notebook/{}".format(name_test)

        predictions_proba, models = utils.load_predict_models(
            dataset, sections, name_models, columns, path_to_save, num_test, index_Xtest=index_Xtest)

        if not os.path.exists("{}/test_{}/summaries".format(path_to_save, num_test)):
                os.makedirs("{}/test_{}/predictions".format(path_to_save, num_test))
                os.makedirs("{}/test_{}/summaries".format(path_to_save, num_test))

        aux = {}
        for section in sections:

            features, scores = summarization.format_data_to_summarize(dataset, section, 'X_test_nf')

            proba_ex1, df_ex1, summaries_ex1, result_ex1 = summarization.pipeline_summarization(
                features, scores, references_df, predictions_proba, section, name_models,
                summ_items, sort_scores=True, proba=True, ascending=False)

            aux[section] = summaries_ex1

            result_ex1.to_csv(
                "{}/test_{}/summaries/{}.csv".format(path_to_save, num_test, section), index=False)
            proba_ex1.to_csv(
                "{}/test_{}/predictions/{}.csv".format(path_to_save,num_test, section), index=False)
            df_ex1.to_csv(
                "{}/test_{}/predictions/df_{}.csv".format(path_to_save, num_test, section), index=False)

        summaries, result_comb = summarization.combine_summaries_eval(
            aux['introduction'], aux['materials'], aux['conclusion'], references_df)

        result_comb.to_csv(
            "{}/test_{}/summaries/{}.csv".format(path_to_save, num_test, 'comb'), index=False)
        
if __name__ == '__main__':
    
    n_test=1
    path_base = "/scratch/cinthiasouza/mv-text-summarizer"
    path_to_read="/scratch/cinthiasouza/mv-text-summarizer/result/{}/{}_*.csv

    
    with open('dataset/dataset_{}.pkl'.format('features'), 'rb') as fp:
        dataset = pickle.load(fp)
        
    references_df = pd.read_csv("dataset/references_df.csv")
    
    columns_name = ['text_rank', 'lex_rank', 'count_one_gram', 'count_two_gram', 'count_three_gram',
        'count_article_keywords',
       'tf-isf', 'position_score', 'paragraph_score',
       'length_score', 'pos_score', 'ner_score', 'dist_centroid']
    
    sections=['introduction', 'materials', 'conclusion']
    name_models = ['knn', 'rf', 'ab', 'gb', 'cb','mlp']
    tests = {'sv_models_embed':'embed'}
    
    columns = list(range(0, 383))
    columns = list(map(str, columns))
    
    Parallel(n_jobs=5)(delayed(main)(
        dataset, tests, num_test, references_df, sections,
        name_models, columns) for num_test in range(1, n_test+1))
 