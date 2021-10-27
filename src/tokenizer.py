import spacy
from bs4 import BeautifulSoup
from pysbd.utils import PySBDFactory


def split_words(sentence, nlp):

  tokens = []
  doc = nlp(str(sentence))
  for token in doc:
    tokens.append(token.text)
  return tokens


def split_sentences(texts):

  nlp = spacy.blank('en')
  nlp.add_pipe(PySBDFactory(nlp))

  sentences = []

  for doc in texts:
    do = nlp(doc.strip())
    sentences.append(list(do.sents))    

  return sentences

def tokenize_paragraph(text):

    soup = BeautifulSoup(text, 'html.parser')
    p = soup.findAll('p')

    return p