from nltk.tokenize import sent_tokenize, word_tokenize
import itertools
import pandas as pd
import numpy as np


def evaluate_summary(candidate, reference, rouge):

    rouge_1 = np.max([rouge.rouge_n(summary=candidate, references=j, n=1) for j in reference])
    rouge_2 = np.max([rouge.rouge_n(summary=candidate, references=j, n=2) for j in reference])
    rouge_l = np.max([rouge.rouge_l(summary=candidate, references=j)for j in reference])
    #nubia_score = max([nubia.score(j, candidate) for j in reference])

    return rouge_1, rouge_2, rouge_l


def score_sentences (candidates, reference, rouge):

    scores = {'rouge_1': [], 'rouge_2': [], 'rouge_l': []}

    for candidate in candidates:

      rouge_1, rouge_2, rouge_l = evaluate_summary(candidate, reference, rouge)

      scores['rouge_1'].append(rouge_1)
      scores['rouge_2'].append(rouge_2)
      scores['rouge_l'].append(rouge_l)
      #scores['nubia'].append(0)

      #if rouge_1 == 0:
      #  scores['nubia'].append(0)
      #  scores['mean'].append( ((rouge_1 + 0)/2) + rouge_2)
      #else:
      #  scores['nubia'].append(nubia_score)
      #  scores['mean'].append( ((rouge_1 + nubia_score)/2) + rouge_2)

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

def main_create_label(candidate, reference, rouge):

  scores = score_sentences(candidate, reference, rouge)

  scores_df = pd.DataFrame(scores)
  label = create_label(scores_df)

  return scores_df, label