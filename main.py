import sys
sys.path.insert(1, '/scratch/cinthiasouza/mv-text-summarizer')

from src import autoencoder
from src import create_embeddings

from src import main_extract_features
from src import main_create_dataset
from src import main_tunning
from src import  main_classifiers
from src import main_summarization

path_base="/scratch/cinthiasouza/mv-text-summarizer/"
path_pp_data='../../pp_data/*'

train_columns = ['text_rank', 'lex_rank', 'count_one_gram', 'count_two_gram',
       'count_three_gram', 'count_article_keywords',
       'tf-isf', 'position_score', 'paragraph_score',
       'length_score', 'pos_score', 'ner_score', 'dist_centroid']
    
under_columns = ['sentences', 'articles', 'text_rank', 'lex_rank', 'count_one_gram', 'count_two_gram',
       'count_three_gram', 'count_article_keywords',
       'tf-isf', 'position_score', 'paragraph_score',
       'length_score', 'pos_score', 'ner_score', 'dist_centroid', 'rouge_1', 'bin']

sections=['introduction', 'materials', 'conclusion']

main_extract_features.main(sections, path_base, path_pp_data)

main_create_dataset.main(
    train_columns, under_columns, sections, path_to_read="mv-text-summarizer/dataset/",
    name_csv="features", format_dataset=True, verbose=True)


with open('{}/dataset_{}.pkl'.format(path_to_read,'features'), 'rb') as fp:
        dataset = pickle.load(fp)
        
path_to_write='../autoencoder_test'

main_autoencoder(
        dataset, sections, X1='X_train_features', X2="X_train_embed", y="y_train",
    path_to_read='dataset', path_to_write=path_to_write, bottleneck_dim=64)

path_to_write

create_embeddings.create_all_embeddings(path_to_read, path_to_write, verbose=False)
            
main_tunning(dataset, X_train='X_train_features', y_train='y_train', sections=sections, path_to_write=path_to_write)
main_tunning(dataset, X_train='X_train_f1', y_train='y_train', sections=sections, path_to_write=path_to_write)
main_tunning(dataset, X_train='X_train_embed', y_train='y_train', sections=sections, path_to_write=path_to_write)

with open('mv_tunning/parameters.pkl', 'rb') as fp:
        parameters = pickle.load(fp)
        
tests = {'sv_models': 'sv', 'sv_models_embed': 'embed' 'mv_models_f1':'mv-f1'}
name_models = ['knn', 'rf', 'ab', 'gb', 'cb',  'mlp']

main_classifiers = main(dataset, parameters, sections, name_models, tests, n_test=1)

references_df = pd.read_csv("dataset/references_df.csv")

columns = list(range(0, 383))
columns = list(map(str, columns))

main(
    dataset, name_models, tests, sections, train_columns, columns, references_df, path_to_write,  n_test=1)