from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import sys
sys.path.insert(1, '/media/cinthia/Dados/Mestrado/mv-text-summarizer')

from sumy.sumy.parsers.plaintext import PlaintextParser
from sumy.sumy.nlp.tokenizers import Tokenizer
from sumy.sumy.summarizers.lsa import LsaSummarizer as SummarizerLsa
from sumy.sumy.summarizers.lex_rank import LexRankSummarizer as SummarizerLex
from sumy.sumy.summarizers.sum_basic import SumBasicSummarizer as SummarizerSumBasic
from sumy.sumy.summarizers.text_rank import TextRankSummarizer  as SummarizerTextrank
from sumy.sumy.nlp.stemmers import Stemmer
from sumy.sumy.utils import get_stop_words

import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import numpy as np
import yake

from src import preprocess
from src import tokenizer



stop_words = list(stopwords.words('english'))
LANGUAGE = "english"
SENTENCES_COUNT=2

def get_citations(text):

  soup = BeautifulSoup(text, 'html.parser')
  bib = soup.findAll('xref')

  return bib


def count_citations(sentences, bibs):

  result = []
  for sentence in sentences:
    count = [len(re.findall(re.escape(str(i)), re.escape(str(sentence)))) for i in bibs]
    result.append(np.sum(count))

  return result

def get_keywords(keywords_article):

  soup = BeautifulSoup(keywords_article, 'html.parser')
  return soup.get_text()

def ner(section, nlp):

  count_ners = []
  ners = []
  for sentences in section:

    doc = nlp(str(sentences))
    ners.append(doc)
    count_ners.append(len(doc.ents))

  return count_ners, ners


def postag(section, nlp):

  tags = []
  pos = []

  for sentences in section:

    count = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}

    doc = nlp(str(sentences))
    count_aux = Counter([token.pos_ for token in doc])
    pos.append(doc)
    
    for i in count.keys():
    
      if count_aux.get(i) == None:
        count[i] = 0
      else:
        count[i] = count_aux.get(i)
    
    tags.append(count.get('NOUN') + count.get('VERB') + count.get("ADJ") + count.get("ADV"))


  return tags, pos


def sentence_len(section, nlp):
  
  sentences_len = [len(tokenizer.split_words(i, nlp)) for i in section]

  return sentences_len

def get_position_score(section):

  position_score = []

  for i in range(len(section) +1):
    
    if i <2 or i == len(section):
      position_score.append(1)
    elif i > 2 and i != len(section) +1:
      position_score.append(1 - ((i-2)/len(section)))
  
  return position_score


def pos_paragraph_score(sentences):

  result = []
  for sentence in sentences:
    if sentence.find("<p id") != -1:
      result.append(1)
    else:
      result.append(0)

  return result

def keywords_yake(text, n=3, lan='en'):

  kw_extractor = yake.KeywordExtractor()
  custom_kw_extractor = yake.KeywordExtractor(lan=lan, n=n, dedupLim=0.9, top=20, features=None)
  keywords = custom_kw_extractor.extract_keywords(text)

  return keywords


def count_keywords(text, keywords):

    n_keywords = []
    for i in text:
        count = 0
        for j in keywords:
            count += len(re.findall(str(j), str(i)))
        
        n_keywords.append(count)

    return n_keywords

def tfisf(text):

  vectorizer = TfidfVectorizer(stop_words=stop_words, analyzer='word', use_idf=True)
  X = vectorizer.fit_transform(text)

  df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
  result = list(df.mean(axis=1))

  return result

def lex_rank(parser):
    
    stemmer = Stemmer(LANGUAGE)
    summarizerLex = SummarizerLex(stemmer)
    summarizerLex.stop_words = get_stop_words(LANGUAGE)

    result = summarizerLex(parser.document, SENTENCES_COUNT)

    summary, sentences = preprocess.format_result(result)

    return summary, sentences

def text_rank(parser):

    stemmer = Stemmer(LANGUAGE)
    summarizertext = SummarizerTextrank(stemmer)
    summarizertext.stop_words = get_stop_words(LANGUAGE)

    result = summarizertext(parser.document, SENTENCES_COUNT)

    summary, sentences = preprocess.format_result(result)

    return summary, sentences

def sentence_embeddings(text, nlp):

  embed = []
  for i in text:
    doc = nlp(i)
    embed.append(doc.vector)
    
  return embed