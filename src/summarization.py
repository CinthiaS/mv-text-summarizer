import numpy as np
import pandas as pd
import re
import json
from sumeval.metrics.rouge import RougeCalculator
rouge = RougeCalculator(stopwords=True, lang="en")

from src import loader
from src import preprocess

def create_label(df, name_model):

    label = [0 for i in range(len(df))]

    j = 0
    for index, row in df.sort_values(name_model, ascending=False).iterrows():
        label[index] = 1
        j +=1

        if j==3:
            break

    return label

def labeling_sentences(X, y, articles, model, name_model, columns_name, scaler):
    
    summaries = []

    for article in articles: 

        x_summ = X.loc[X['articles'] == article]
        y_summ = y.loc[y['articles'] == article]

        X_test = x_summ[columns_name]
        y_test = y_summ['rouge_1']*100

        X_test = scaler.transform(X_test)

        y_pred= model.predict(X_test)

        summ_df = pd.DataFrame({'sentences': x_summ['sentences'], 'rouge_1': list(y_test),
                            'predictions' : y_pred.reshape(1, -1)[0], 'articles': list(x_summ['articles'])})

        summ_df.reset_index(inplace=True, drop=True)
        summ_df[name_model] = create_label(summ_df, 'predictions')

        summaries.append(summ_df)
    
    df = pd.concat(summaries)
    
    return df

def get_ref_summary(file_name, path_base):

    text, files = loader.load_files(path_base, [file_name])
    reference = preprocess.format_xml(text[0].get('sec_abstract'))
    reference = re.sub('(?<=<title>)(.*?)(?=</title>)', '', reference)
    reference = re.sub(r'[\t\n\r]', '', reference)
    reference = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", reference)
    reference = preprocess.xml_to_text(reference)
    reference = preprocess.format_text(str(reference), post_processing=True)
    reference = re.sub(r' +', " ", reference)
    reference = reference.strip()
    
    return reference

def create_summaries(df, references, articles, name_models):

    df_summaries = pd.DataFrame()
    
    for name_model in name_models:
        
        summaries = []

        for article in articles:
        
            try: 
                text = df.loc[df['articles'] == article]
                summaries.append(' '.join(text.loc[text[name_model] == 1]['sentences'].values))
            except TypeError:
                summaries.append("")
        
        
        df_summaries[name_model] = summaries
    
    df_summaries['references'] = references
     
    return  df_summaries

def combine_summ(summaries_intro, summaries_mat, summaries_conc, references, name_models):
    summaries_comb = pd.DataFrame()

    for name_model in name_models:
    
        summaries_comb[name_model] = summaries_intro[name_model] + summaries_mat[name_model] + summaries_conc[name_model]
    
    summaries_comb['references'] = references
    return summaries_comb    

def create_df(name_models, x_summ, y_true, predictions, section, proba=False):
    
    df = pd.DataFrame({'sentences': x_summ['sentences'],
                       'rouge_1': list(y_true),
                       'articles': x_summ['articles']})
    
    if proba == False:
        for name_model in name_models:
            df[name_model] = predictions[section][name_model].reshape(1, -1)[0]
    else:
        for name_model in name_models:
            y_pred = [i[1] for i in predictions[section][name_model]]
            df[name_model] = y_pred
        
    return df
        
def rouge_metrics(candidate, reference):

    rouge_1 = rouge.rouge_n(summary=candidate, references=reference, n=1)
    rouge_2 = rouge.rouge_n(summary=candidate, references=reference, n=2)
    rouge_l = rouge.rouge_l(summary=candidate, references=reference)

    return rouge_1, rouge_2, rouge_l

def evaluate_summaries(df, name_models):

    vfunc = np.vectorize(rouge_metrics)
    
    for name_model in name_models:
        
        df['{}_r1'.format(name_model)],df['{}_r2'.format(name_model)],df['{}_rl'.format(name_model)] = vfunc(df[name_model], df['references'])
        
    return df
