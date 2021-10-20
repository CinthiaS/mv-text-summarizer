import pandas as pd
import numpy as np

from src import summarization

def create_df(name_model, x_summ, y_true, y_pred):
    
    df = pd.DataFrame({'sentences': x_summ['sentences'],
                       'rouge_1': list(y_true),
                        name_model : y_pred.reshape(1, -1)[0],
                       'articles': x_summ['articles']})
        
    return df

def main(name_model, intro_result, mat_result, conc_result, summ_items, path_base):

    references = [summarization.get_ref_summary(i, path_base) for i in summ_items]
    
    summaries = summarization.create_summaries_df(name_model, intro_result, mat_result, conc_result, summ_items, references)
        
    summaries = summarization.evaluate_summaries(summaries, sections=['intro', 'mat', 'conc', 'concat'])
    
    return summaries