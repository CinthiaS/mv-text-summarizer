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

def create_summaries(df, articles, name_model):

    summaries = []

    for article in articles:
        
        text = df.loc[df['articles'] == article]
        summaries.append(' '.join(text.loc[text[name_model] == 1]['sentences'].values))
     
    return  summaries

def create_summaries_df(name_model, intro_model, mat_model, conc_model, summ_items, references):
    
    summaries_intro = create_summaries(intro_model, summ_items, '{}_intro'.format(name_model))
    summaries_mat = create_summaries(mat_model, summ_items, '{}_mat'.format(name_model))
    summaries_conc = create_summaries(conc_model, summ_items, '{}_conc'.format(name_model))
    

    summaries = pd.DataFrame({'articles': summ_items,
                             'references': references,
                             'summaries_intro': summaries_intro,
                             'summaries_mat': summaries_mat,
                             'summaries_conc':summaries_conc})

    summaries["summaries_comb"] = summaries['summaries_intro'] + " " + summaries['summaries_mat']  + " " +  summaries['summaries_conc']
    summaries.to_csv("summaries_{}.csv".format(name_model), index=False)
        
    return summaries
        
def rouge_metrics(candidate, reference):

    rouge_1 = rouge.rouge_n(summary=candidate, references=reference, n=1)
    rouge_2 = rouge.rouge_n(summary=candidate, references=reference, n=2)
    rouge_l = rouge.rouge_l(summary=candidate, references=reference)

    return rouge_1, rouge_2, rouge_l

def evaluate_summaries(df, sections):

    vfunc = np.vectorize(rouge_metrics)
    
    for i in sections:
        
        df['{}_r1'.format(i)],df['{}_r2'.format(i)],df['{}_rl'.format(i)] = vfunc(df['summaries_{}'.format(i)], df['references'])
    
    return df

def pipeline_summarization(name_model, models, dataset, columns_name, summ_items, path_base, sections=['intro', 'mat', 'conc', 'concat']):
    
    results = []
    
    for i in dataset.keys():
        
        X = dataset.get(i)[1]
        y = dataset.get(i)[3]
        
        results.append (labeling_sentences(
        X, y, summ_items, model=models[i], name_model='{}_{}'.format(name_model, i),
        columns_name=columns_name))


    references = [get_ref_summary(i, path_base) for i in summ_items]
    
    summaries = create_summaries_df(name_model, results[0], results[1], results[2], summ_items, references)
    
    summaries = evaluate_summaries(summaries, sections=sections)
    
    return summaries, results[0], results[1], results[2]