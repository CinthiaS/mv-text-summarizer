
  
import pandas as pd
from sklearn.cluster import DBSCAN

from sumy.sumy.parsers.plaintext import PlaintextParser
from sumy.sumy.nlp.tokenizers import Tokenizer

from nltk.corpus import stopwords

from src import extract_features
from src import preprocess
from src import utils
from src import cluster_analyzer
from src import tokenizer

stop_words = list(stopwords.words('english'))
LANGUAGE = "english"

def main(df, nlp_sm, nlp_md):
    
    parser = PlaintextParser(". ".join(list(df['sentences'])), Tokenizer(LANGUAGE))
    _ , sentences_text = extract_features.text_rank(parser)
    _ , sentences_lex = extract_features.lex_rank(parser)
    _, sentences_lsa = extract_features.lsa_rank(parser)

    summ = {'lsa_rank': sentences_lsa,  'text_rank': sentences_text, 'lex_rank':sentences_lex}
    df = pd.DataFrame(summ).reset_index()
    df = df.rename(columns={'index': 'sentences'})

    df['count_postag'] = df['sentences'].apply(extract_features.count_postag)
    df['count_ner'] = df['sentences'].apply(extract_features.count_ner)
    df['sentence_len'] = df['sentences'].apply(extract_features.score_len)

    text_noise = [preprocess.remove_noise(i) for i in list(df['sentences'])]
    pp_text = preprocess.stemming(text_noise, nlp_sm, stop_words)

    one_gram = extract_features.keywords_yake(pp_text, n=1, lan='en')
    two_gram = extract_features.keywords_yake(pp_text, n=2, lan='en')
    three_gram = extract_features.keywords_yake(pp_text, n=3, lan='en')

    one_gram = [key for key, _ in one_gram]
    two_gram = [key for key, _ in two_gram if len(key.split(' ')) > 1]
    three_gram = [key for key, _ in three_gram if len(key.split(' ')) > 2]

    df['count_one_gram'] = df['sentences'].apply(extract_features.count_keywords, args=(one_gram,))
    df['count_two_gram'] = df['sentences'].apply(extract_features.count_keywords, args=(two_gram,))
    df['count_three_gram'] = df['sentences'].apply(extract_features.count_keywords, args=(three_gram,))

    df['tfisf'] = extract_features.tfisf(list(df['sentences']))

    df['position_score'] = extract_features.get_position_score(list(df['sentences']))

    embed = extract_features.sentence_embeddings(list(df['sentences']), nlp_md)
    df_embed = pd.DataFrame(embed)

    clustering = DBSCAN(eps=2, min_samples=2).fit(embed)
    cluster_df = cluster_analyzer.cluster_analisys(df_embed, clustering, normalize=False, verbose=False)
    
    
    df_embed['sentences'] = list(df['sentences'])

    df['dist_centroid'] = cluster_df['dis_centroid']

    return df, df_embed
