from nltk.tokenize import sent_tokenize, word_tokenize
import itertools
import pandas as pd
import numpy as np

import rouge

evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
                           limit_length=False,
                           #length_limit_type='words',
                           apply_avg=False,
                           apply_best=True,
                           alpha=0.5,
                           weight_factor=1.2,
                           stemming=True)

def evaluate_summary(candidate, reference):
    
    r1 = []
    r2 = []
    rl = [] 
    
    for r in reference:
        scores = evaluator.get_scores(hypothesis=candidate, references=r) 
    
        r1.append(scores['rouge-1']['f'])
        r2.append(scores['rouge-2']['f'])
        rl.append(scores['rouge-l']['f'])

    return [np.max(r1), np.max(r2), np.max(rl)]

def score_sentences (candidates, reference, rouge):

    scores = {'rouge_1': [], 'rouge_2': [], 'rouge_l': []}

    for candidate in candidates:

      rouge_1, rouge_2, rouge_l = evaluate_summary(candidate, reference)

      scores['rouge_1'].append(rouge_1)
      scores['rouge_2'].append(rouge_2)
      scores['rouge_l'].append(rouge_l)

    return scores


def create_label(scores_df):

    label = [0 for i in range(len(scores_df))]

    j = 0
    for index, row in scores_df.sort_values('rouge_2', ascending=False).iterrows():
      label[index] = 1
      j +=1

      if j==3:
        break

    return label

def main_create_label(df, reference, rouge):

    df['rouges'] = df['sentences'].apply(evaluate_summary, args=(reference,))

    return df