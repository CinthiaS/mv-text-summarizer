import re
import tokenizer
from nltk.stem import PorterStemmer
import extract_features

def remove_noise(text):

  text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
  text = re.sub(r'\([^)]*\)', '', text)
  text = re.sub('[^A-Za-z0-9]+', ' ', text)
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

def format_text(text, post_processing=False):

  text = text.replace(".<xref", ". <xref")
  text = text.replace("</p>","</p> ")
  text = text.replace('.</p>', "</p>.")
  if post_processing:
    text = text.replace("-", " ")
    text = text.replace("–", '')
    text = text.replace("(,)", "")
    text = text.replace("()", "")
    text = text.replace("[,]", "")
    text = text.replace("[]", "")
    text = text.replace("(; )", "")
    text = text.replace("(; )", "")

  return text

def format_sentences_xml(sentences_xml):

  aux = []

  for i in sentences_xml:
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

def exclude_bib(text, bibs):

  for i in bibs:
    text = text.replace(str(i), '')
    
  return text

def format_article_keywords(keywords_article, number_text ):

  keys = extract_features.get_keywords(keywords_article[number_text])
  keys = keys.split("\n")
  keys = list(filter(None, keys))

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