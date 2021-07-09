import spacy
from pysbd.utils import PySBDFactory
from bs4 import BeautifulSoup


def split_words(sentence, nlp):

  tokens = []
  doc = nlp(str(sentence))
  for token in doc:
    tokens.append(token.text)
  return tokens

def split_sentences(texts):

  nlp = spacy.blank('en')
  nlp.add_pipe(PySBDFactory(nlp))

  sent_intro = []

  for doc in texts:
    do = nlp(doc.strip())
    sent_intro.append(list(do.sents))     

  return sent_intro

def tokenize_paragraph(text):

    soup = BeautifulSoup(text, 'html.parser')
    p = soup.findAll('p')

    return p