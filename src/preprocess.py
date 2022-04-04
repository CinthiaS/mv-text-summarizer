import re

from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from src import tokenizer
from src import extract_features

def xml_to_text(text):

    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()

    return text

def remove_unicode(text):

    text = str(text).encode("ascii", "ignore")
    text = text.decode()

    return text

def remove_noise(text):

  text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
  text = re.sub(r'\([^)]*\)', '', text)
  text = re.sub('(?<=<title>)(.*?)(?=</title>)', '', text)
  text = re.sub('[^A-Za-z0-9]+', ' ', text)
  text = re.sub(r'[\t\n\r]', '', text)
  text = str(text.encode('ascii', 'ignore'))

  text = text.lower()

  return text

def stemming(text, nlp, stop_words):

  ps = PorterStemmer()

  text = " ".join(text).lower()
  words = tokenizer.split_words(text, nlp)
  words = [ps.stem(w) for w in words  if not w in stop_words]

  return " ".join(words)

def format_intro(text):

  text = text.replace("INTRODUCTION", "")
  text = text.replace("Introduction", "")
  text = text.replace('\n\nOBJECTIVE\n', '')
  text = text.replace('\n\nObjectives\n', '')
  text = text.replace('\nSummary\n\n', '')
  text = text.replace("\n", "")

  return text

def format_xml(xml):

  xml = xml.replace(".<xref", ". <xref")
  xml = xml.replace("</p>","</p>  " )
  xml = xml.replace('.</p>', "</p>.")
  xml = xml.replace('<title-introduction><title></title>', '')
  xml = xml.replace('</title-introduction>', '')
  xml = xml.replace("<italic>et al</italic>.", "<italic>et al</italic>")

  return xml

def clean_text(text):

    text = re.sub(r'\@xcite', ' ', text)
    text = re.sub(r'\[[^()]*\]', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text

def format_text(text, post_processing=False):

  text = text.replace(".<xref", ". <xref")
  text = text.replace("</p>","</p> ")
  text = text.replace('.</p>', "</p>.")
  if post_processing == False:
    text = re.sub('(?<=<title>)(.*?)(?=</title>)', '', text)


  if post_processing:
    text = text.replace("-", " ")
    text = text.replace("–", '')
    text = text.replace("(,)", "")
    text = text.replace("()", "")
    text = text.replace("[,]", "")
    text = text.replace("[]", "")
    text = text.replace("(;)", "")
    text = text.replace("(; )", "")
    text = re.sub(r' +', " ", text)
    text = str(text.encode('ascii', 'ignore'))
    
    text = text.strip()

  return text

def format_sentences_xml(sentences_xml):

  aux = []

  for i in sentences_xml:
    sentence = xml_to_text(i)
    if not sentence.isspace() and  sentence != '' :
      if i.find("<p id=") != -1:
        new_sentences = i.split("  ")

        for j in new_sentences:
          aux.append(j)
      else:
        aux.append(i)

  return aux


def format_sentences(sentences):

  sentences = [re.sub(r'^, +',' ', sentence).strip() for sentence in sentences]

  return sentences


def format_article_keywords(keywords_article ):

  keys = extract_features.get_keywords(keywords_article)
  keys = keys.split("\n")
  keys = list(filter(None, keys))
  keys =[remove_noise(i) for i in keys]

  last_key = keys[-1].split(' ')
  if last_key[0] == 'and':
    keys[-1] = " ".join(last_key[1:])

  return keys


def format_result(result):

  summary = ' '.join(list(map(str, list(result[0]))))

  aux = {}
  for key, value in result[1].items():
    aux [str(key)] = float(value)

  return summary, aux


def replace_bib(text, bibs):

  for i in bibs:
    text = text.replace(str(i), '')
    
  return text

def remove_citations(xml, text):
  
  bibs = extract_features.get_citations(xml)
  text = replace_bib(text, bibs)
  text = format_text(text, post_processing=True)
  
  return text