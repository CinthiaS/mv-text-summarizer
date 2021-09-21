import re
import pandas as pd
import json
from sumeval.metrics.rouge import RougeCalculator
rouge = RougeCalculator(stopwords=True, lang="en")
from sklearn.preprocessing import StandardScaler

import preprocess

def create_label(df):

    label = [0 for i in range(len(df))]

    j = 0
    for index, row in df.sort_values('predictions', ascending=False).iterrows():
      label[index] = 1
      j +=1

      if j==3:
        break

    return label

def evaluate_summaries(candidate, reference):

    rouge_1 = rouge.rouge_n(summary=candidate, references=reference, n=1)
    rouge_2 = rouge.rouge_n(summary=candidate, references=reference, n=2)
    rouge_l = rouge.rouge_l(summary=candidate, references=reference)

    return [rouge_1, rouge_2, rouge_l]

def summarize(X_val, y_val, scaler, model):

        columns_name = ['text_rank', 'lex_rank', 'count_one_gram', 'count_two_gram',
                'count_three_gram', 'count_article_keywords', 'tf-isf',
                'position_score', 'paragraph_score', 'number_citations', 'length_score',
                'pos_score', 'ner_score', 'dist_centroid']

        #sum_x_test = X_val.loc[X_val['number_text'] == n]
        try:
                sentences = X_val[0]
        except KeyError:
                sentences = X_val['0']
                
        sum_x_test = X_val[columns_name]
        sum_y_test = y_val['rouge_1']*100

        sum_x_test = scaler.transform(sum_x_test)
        y_pred = model.predict(sum_x_test)

        summ_df = pd.DataFrame({'sentences': sentences, 'predictions': y_pred.reshape(1, -1)[0]})
        summ_df.reset_index(inplace=True, drop=True)
        summ_df['label'] = create_label(summ_df)
        summary = ' '.join(summ_df.loc[summ_df['label'] == 1]['sentences'].values)

        return summ_df, summary
    
def get_ref_summary(file_name):

    text = loader.load_files(path_base, [file_name])
    reference = preprocess.format_xml(text[0].get('sec_abstract'))
    reference = re.sub('(?<=<title>)(.*?)(?=</title>)', '', reference)
    reference = re.sub(r'[\t\n\r]', '', reference)
    reference = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", reference)
    reference = preprocess.xml_to_text(reference)
    reference = preprocess.format_text(str(reference), post_processing=True)
    reference = re.sub(r' +', " ", reference)
    reference = reference.strip()
    
    return reference

def load_summarize_models(model, dataset):

    if model == 'mlp':
        json_file = open('model_{}.json'.format(dataset), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('model_{}.h5'.format(dataset))
    else: 
        infile = open('{}_{}'.format(model, dataset),'rb')
        model = pickle.load(infile)
        infile.close()

    infile = open('StandardScaler_{}'.format(dataset),'rb')
    scaler = pickle.load(infile)
    infile.close()
    
    return model, scaler

def summarizer_all_texts(model_intro, scaler_intro, model_mat, scaler_mat, model_conc, scaler_conc, path):

    summaries_intro = []
    summaries_mat = []
    summaries_conc = []
    summaries_merged = []
    references = []

    files = glob.glob(path)
    files = [i.split("/")[-1] for i in files]

    for file in files:

        name = file.split('_')[1]

        X_intro = pd.read_csv("../sumdata/{}/{}".format('introduction', file))
        y_intro =  pd.read_csv("../sumdata/{}/scores_{}".format('introduction', name))

        X_mat = pd.read_csv("../sumdata/{}/{}".format('materials', file))
        y_mat =  pd.read_csv("../sumdata/{}/scores_{}".format('materials', name))

        X_conc = pd.read_csv("../sumdata/{}/{}".format('conclusion', file))
        y_conc =  pd.read_csv("../sumdata/{}/scores_{}".format('conclusion', name))

        _, summary_intro = summarize(X_intro, y_intro, scaler_intro, model_intro)
        _, summary_mat = summarize(X_mat, y_mat, scaler_mat, model_mat)
        _, summary_conc = summarize(X_conc, y_conc, scaler_conc, model_conc)
        merged = summary_intro + " " + summary_mat + " " + summary_conc

        summaries_intro.append(summary_intro)
        summaries_mat.append(summary_mat)
        summaries_conc.append(summary_conc)
        summaries_merged.append(merged)

        name_articles = name.replace(".csv", ".json")

        references.append(get_ref_summary(name_articles))


    names = [i.split('_')[1].replace(".csv", "") for i in files]
    summaries = pd.DataFrame({"article": names, "reference": references, "intro": summaries_intro, "mat": summaries_mat,"conc": summaries_conc,"merged": summaries_merged})

    return summaries

  def evaluation_all_summaries(summaries):

    all_scores = []
    i = 0
    for index, row in summaries.iterrows():

        scores_intro = evaluate_summaries(row['intro'], row['reference'])
        scores_mat = evaluate_summaries(row['mat'], row['reference'])
        scores_conc = evaluate_summaries(row['conc'], row['reference'])
        scores_merged = evaluate_summaries(row['merged'], row['reference'])

        scores = pd.DataFrame({
                            "article": row['article'],
                            "intro_r1": scores_intro[0],
                            "intro_r2": scores_intro[1], 
                            "intro_r3": scores_intro[2],  
                            "mat_r1": scores_mat[0],
                            "mat_r2": scores_mat[1],
                            "mat_rl": scores_mat[2],
                            "conc_r1": scores_conc[0],
                            "conc_r2": scores_conc[1],
                            "conc_rl": scores_conc[2],
                            "merged_r1": scores_merged[0],
                            "merged_r2": scores_merged[1],
                            "merged_rl": scores_merged[2]},  index=[i])

        i+=1

        all_scores.append(scores)

    df = pd.concat(all_scores)

    return df