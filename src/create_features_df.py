import pandas as pd

import extract_features
import preprocess
import utils
import cluster_analyzer
import tokenizer

from sklearn.cluster import DBSCAN

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))

LANGUAGE = "english"

def format_df (
    text, sentences_text, sentences_lex, count_one_gram, count_two_gram, count_three_gram, count_article_keywords, result_tfisf, position_score, paragraph_score, number_citations, length_score, pos_score, ner_score, cluster_df):

  scores = []
  data = {i: [] for i, j in sentences_text.items()}

  for key, value in sentences_text.items():
    
    data[key].append(value)
    data[key].append(sentences_lex.get(key))
    data[key].append(count_one_gram[text.index(key)])
    data[key].append(count_two_gram[text.index(key)])
    data[key].append(count_three_gram[text.index(key)])
    data[key].append(count_article_keywords[text.index(key)])
    data[key].append(result_tfisf[text.index(key)])
    data[key].append(position_score[text.index(key)])
    data[key].append(paragraph_score[text.index(key)])
    data[key].append(number_citations[text.index(key)])
    data[key].append(length_score[text.index(key)])
    data[key].append(pos_score[text.index(key)])
    data[key].append(ner_score[text.index(key)])

    try:
      data[key].append(cluster_df.loc[cluster_df['sentence'] == text.index(key)]['dis_centroid'].values[0])
    except IndexError:
      data[key].append(0)

  columns = ['sentences', 'text_rank', 'lex_rank', 'count_one_gram', 'count_two_gram', 'count_three_gram', 'count_article_keywords', "tf-isf", 'position_score', 'paragraph_score', 'number_citations', 'length_score', 'pos_score', 'ner_score', 'dist_centroid']
  features = pd.DataFrame(data.items())
  scores = pd.DataFrame(features[1].to_list(), columns=columns[1:])
  del features[1]
  features = pd.concat([features, scores], axis=1)

  return features


def main_features_df(text, xml, article_keywords, number_text, nlp_sm, nlp_md):

  sentences = list(map(str, text))
  
  #Count citations
  bibs = extract_features.get_citations(xml)
  sentences_xml = tokenizer.split_sentences([xml])
  sentences_xml = list(map(str, sentences_xml[0]))
  sentences_xml = preprocess.format_sentences_xml(sentences_xml)
  number_citations = extract_features.count_citations(sentences_xml, bibs)

  if len(number_citations) != len(sentences):
    print(len(number_citations))
    print(len(sentences))
    raise ValueError('VectorSize')

  #paragraph position score
  paragraph_score = extract_features.pos_paragraph_score(sentences_xml)

  #POS
  count_tag, pos = extract_features.postag(text, nlp_sm)
  count_tag = [utils.check(i) for i in count_tag ]
  pos_score = count_tag #calc_score(count_tag, metric='max')

  #NER
  count_ner, ners = extract_features.ner(text, nlp_sm)
  ner_score = count_ner #calc_score(count_ner, metric='max')

  # Position score
  sent_len = extract_features.sentence_len(sentences, nlp_sm)
  length_score = sent_len #calc_score(sent_len, metric='max')

  position_score = extract_features.get_position_score(sentences)

  #Keywords
  text_noise = [preprocess.remove_noise(i) for i in sentences]
  pp_text = preprocess.stemming(text_noise, nlp_sm, stop_words)

  one_gram = extract_features.keywords_yake(pp_text, n=1, lan='en')
  two_gram = extract_features.keywords_yake(pp_text, n=2, lan='en')
  three_gram = extract_features.keywords_yake(pp_text, n=3, lan='en')

  one_gram = [key for key, _ in one_gram]
  two_gram = [key for key, _ in two_gram if len(key.split(' ')) > 1]
  three_gram = [key for key, _ in three_gram if len(key.split(' ')) > 2]

  count_one_gram = extract_features.count_keywords(text, one_gram)
  count_two_gram = extract_features.count_keywords(text, two_gram)
  count_three_gram = extract_features.count_keywords(text, three_gram)

  ngrams = [count_one_gram, count_two_gram, count_three_gram ]

  #article keywords
  article_keywords_list = preprocess.format_article_keywords(article_keywords, number_text)
  count_article_keywords = extract_features.count_keywords(text, article_keywords_list)

  #TF-ISF
  result_tfisf = extract_features.tfisf(text)

  #LexRank score
  parser = PlaintextParser(" ".join(sentences), Tokenizer(LANGUAGE))
  summary_text, sentences_text = extract_features.text_rank(parser)

  #TextRank score
  parser = PlaintextParser(" ".join(sentences), Tokenizer(LANGUAGE))
  summary_lex, sentences_lex = extract_features.lex_rank(parser)

  #Embeddings
  text = [preprocess.remove_noise(i) for i in text]
  embed = extract_features.sentence_embeddings(text_noise, nlp_md)
  df_embed = pd.DataFrame(embed)

  #Clustering 
  clustering = DBSCAN(eps=2, min_samples=2).fit(embed)
  cluster_df = cluster_analyzer.cluster_analisys(df_embed, clustering, normalize=True, verbose=False)

  features = {'pos_score': pos_score, 'pos': pos, 'ner_score': ner_score, 'ners': ners,
                'position_score': position_score, 'number_citations':number_citations,
                'paragraph_score': paragraph_score,  'length_score': length_score, 'ngrams': ngrams,
                'count_article_keywords': count_article_keywords, 'summary_text':summary_text,
                'sentences_text':sentences_text, 'summary_lex': summary_lex, 'sentences_lex':sentences_lex,
                'cluster_df': cluster_df,  'result_tfisf':result_tfisf }

  return features
