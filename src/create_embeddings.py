import pickle
import json
import pandas as pd
import glob

#from sentence_transformers import SentenceTransformer
#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_embeddings(sentences):
    
    embeddings = model.encode(sentences)
    embedding_list = [embedding for sentence, embedding in zip(sentences['summary'], embeddings)]

def create_all_embeddings(dataset, sections, path_to_read, path_to_write, generate=False, verbose=False):
        
    if generate:
        for section in sections:
    
            sentences_train = dataset[section]['X_train_nf']
            sentences_test = dataset[section]['X_train_nf']
    
            embedding_list = get_embeddings(sentences_train['sentences'].tolist())
            dataset[section]['X_train_embed'] = pd.DataFrame(embedding_list)
        
            embedding_list = get_embeddings(sentences_test['sentences'].tolist())
            dataset[section]['X_test_embed'] = pd.DataFrame(embedding_list)
            
    else:
        
        for i in glob.glob('{}/embed*.csv'.format(path_to_read)):
            for section in sections:
                if i.find(section) != 0:
                    if i.find('train'):
                        dataset[section]['X_train_embed']  = pd.read_csv(i)
                    elif i.find('test'):
                        dataset[section]['X_test_embed']  = pd.read_csv(i)
    
    if verbose:
        print("Write dataset")
    
    with open('{}/dataset_{}.pkl'.format(path_to_write, 'features'), 'wb') as fp:
        pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
