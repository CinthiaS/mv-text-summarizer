#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 20:19:22 2021

@author: Cinthia M. Souza
"""

import os
from bs4 import BeautifulSoup
import json
import re
import subprocess
import shutil
import create_json
import argparse
import clean_xml

def get_title(soup):

  try:
    title = soup.find('article-title')
  except AttributeError:
    title = float('nan')

  return title

def get_title_section(soup):

  try:
    title_sections = soup.findall('title')
  except AttributeError:
    title_sections = float('nan')

  return title_sections

def get_keywords(soup):

  try:
    keywords = soup.find('kwd-group')
  except AttributeError:
    keywords = float('nan')

  return keywords

def get_abstract(soup):

  try:
    abstract = soup.find('abstract')
  except AttributeError:
    abstract = float('nan')

  return abstract

def get_name_sections(soup, text=False):

    sections = []
    for i in soup.find_all('title'):
        try:
            if text:
                text = i.get_text()
                sections.append(text)
            else:
                sections.append(i)
        except ValueError:
            pass

    return sections

def remove_noise(text):

  text = re.sub(r'<title>.*?</title>', '', text)
  text = re.sub(r'<italic>.*?</italic>', '', text)
  text = re.sub(r'[^\x00-\x7F]+', '', text)
  text = text.replace("\n"," ")
  text = text.replace("\\n"," ")
  text = re.sub(r' +', ' ', text)
  text = text.lower()
  text = text.strip()

  return text

def remove_noise_keys(text):

  #Remove "\n" da string
  text = text.replace("\n"," ")
  text = text.replace("\\n"," ")
  text =  re.sub(r"\b(?:[A-Z]\.)+(?!\w)",'', text)
  text = re.sub('^([IVXLCM]+)\\.?$', '', text)
  text = re.sub('IV. ', '', text)
  text = re.sub('V. ', '', text)
  text = re.sub('VI. ', '', text)
  text = re.sub('VII. ', '', text)
  text = re.sub('VIII. ', '', text)
  text = re.sub('IX. ', '', text)
  text = re.sub('X. ', '', text)
  text = re.sub('III. ', '', text)
  text = re.sub('II. ', '', text)
  text = re.sub('I. ', '', text)
  text = text.replace(re.escape('/'), ' ')
  text = text.replace('\-', ' ')
  text = text.replace(':', "")
  text = re.sub(r'/[^a-zA-Z0-9]/g', '', text)
  text = text.lower()
  #text = re.sub('(.)\\1{1,}', '',text)
  text = re.sub(r'(?<!\w)([a-z])\.', '', text)
  text = re.sub(r'((\d+)[\.])(?!([\d]+))', '', text)
  text = re.sub(r"\b\d+\b", "", text)
  text = re.sub(r'\([^)]*\)', '', text)
  text = re.sub('\) ', '', text)
  text = text.replace(".","")
  text = text.strip()

  return text

def format_abstract(text):

  text = remove_noise(text)
  text = "<abstract>{}</abstract>\n<body>".format(text)

  return text

def rewrite_xml(file_xml, name_file, sections): 

    new_tags = []
    
    j = 0
    for i in sections:

        if j == 0:
            new_tag = "<title-{}>\n {}".format(i.get_text().replace(":", "").replace(" ", "-").lower(), i)
        else:
            new_tag = "</title-{}>\n <title-{}>\n {}".format(sections[j-1].get_text().replace(":", "").replace(" ", "-").lower(), i.get_text().replace(":", "").replace(" ", "-").lower(), i)

        
        new_tags.append(new_tag)
        file_xml = file_xml.replace(str(i), new_tag)
        j+=1

    return file_xml

def text_segmentation_single(soup):

    title = get_title(soup)
    keywords = get_keywords(soup)
    abstract = get_abstract(soup)
    
    return title, keywords, abstract

def remove_tag(soup, tag_name):

  front = soup.find_all(tag_name)
  for f in front:
    f.extract()

  return soup

def refactoring_name_sections(soup, sections_text, sections_tags):

  text = str(soup)

  sections_text = [remove_noise_keys(i) for i in sections_text]

  i = 0
  for name in sections_text:
    text=text.replace(str(sections_tags[i]), "\n\n<title>{}</title>".format(name))
    i+=1

  soup  = BeautifulSoup(text, features="lxml", from_encoding='latin-1')

  return soup

def main(dataset):

  texts = os.listdir(dataset)
  
  aux =  os.listdir("{}_pp".format(dataset))
  cont = len(aux) + 1
  aux = [i.replace('.json', '.xml') for i in aux]
  texts = list(set(texts) - set(aux))

  for file_name in texts:

    # print(file_name)

    data = {}

    text = open("{}/{}".format(dataset, file_name), encoding='latin-1').read()

    try: 
      text = clean_xml.main(text)
    except RecursionError:
      pass

    if (text.lower().find("<title>introduction</title>") == -1) and (text.lower().find("<title>1. introduction</title>") == -1) and (text.lower().find("<title>i. introduction</title>") == -1):
      text = text.replace("<body>", "<body>\n<title>introduction</title>")

    soup  = BeautifulSoup(text, features="lxml", from_encoding='latin-1')
    
    try:
   
      title_article = get_title(soup).get_text()
      keywords = get_keywords(soup)

      abstract = get_abstract(soup).get_text()
      abstract = remove_noise(abstract)

      journal_title = soup.find("journal-title")
      id = soup.find("article-id", {'pub-id-type':'pmid'}).get_text()

      try:
        doi = soup.find("article-id", {'pub-id-type':'doi'}).get_text()
      except AttributeError:
        doi = "None"

      soup = remove_tag(soup, tag_name='front')
      try:
        text = clean_xml.main(str(soup))
      except RecursionError:
        pass
      soup = BeautifulSoup (text, features="lxml")

      sections_text = get_name_sections(soup, text=True)
      sections_tags = get_name_sections(soup, text=False)
      
      soup = refactoring_name_sections(soup, sections_text, sections_tags)

      sections_text = get_name_sections(soup, text=True)
      sections_tags = get_name_sections(soup, text=False)

      text = rewrite_xml(str(soup), file_name, sections_tags)
      soup  = BeautifulSoup(text, features="lxml")
        
      if(sections_text != []):
      
          data['journal_title'] = str(journal_title.get_text())
          data['id'] = str(id)
          data['doi'] = str(doi)
          data['title'] = str(title_article)
          data['keywords'] = str(keywords)
          data['abstract'] = str(abstract)
          data['title_sections'] = list(map(str,sections_text))
          
          try:
            for j in sections_text:
              data[j] = " "
            for j in sections_text:
              data[j] +=  str(soup.find_all("title-{}".format(j.replace(":", "").replace(" ", "-").lower()))[0]) + " "

            data = create_json.main(data)
            
            if data != None:
              with open('{}_pp/'.format(dataset) + str(file_name).replace('.xml', '') + '.json', 'w') as file_json:
                json.dump(data, file_json)
              cont+=1

            if cont % 2000 == 0:
              print("Número de arquivos salvos: {}".format(cont))

          except TypeError:
            pass
          except IndexError:
            pass
    except AttributeError:
      pass
    except IndexError:
      pass
    except TypeError:
      pass

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_name', "-d",required=True)
     
  args = parser.parse_args()

  if not os.path.exists("{}_pp".format(args.dataset_name)):
    os.makedirs("{}_pp".format(args.dataset_name))

  main(args.dataset_name)
