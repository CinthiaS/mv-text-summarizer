import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from collections import Counter

import seaborn as sns
#sns.set_theme(style="whitegrid")

def create_label(scores_df):

    label = [0 for i in range(len(scores_df))]

    j = 0
    for index, row in scores_df.sort_values('rouge_1', ascending=False).iterrows():
        label[index] = 1
        j +=1

        if j==3:
            break

    return label

def transform_to_classification(df):

    articles = np.unique(df['articles'])
    labels = []
    articles_list = []
    for article in articles:

        y = df.loc[df['articles'] == article].reset_index(drop=True)

        articles_list.append(y['articles'])
        y['bin'] = create_label(y)
        labels.append(y)

    return pd.concat(labels)

def shuffle_dataset(X, y):
    
    idx = np.random.permutation(len(X))

    X = X[idx]
    y = np.array(y)[idx]
    
    return X, y

def bar_plot(y):
    
    count = Counter(y)
    df = pd.DataFrame.from_records(list(dict(count).items()), columns=['label','count'])

    ax = sns.barplot(x="label", y="count", data=df)
    
def load_keras_model(path_to_save, name_model, section, num_test):

    json_file = open('{}/test_{}/{}_{}.json'.format(path_to_save, num_test, name_model, section), 'r')
    model = json_file.read()
    json_file.close()
    model = model_from_json(model)
    model.load_weights('{}/test_{}/{}_{}.h5'.format(path_to_save, num_test, name_model, section))
    print("Loaded model from disk")
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(
                            learning_rate=0.001), metrics=[keras.metrics.Precision()])
    
    return model

def load_predict_models(dataset, sections, name_models, columns, path_to_save, num_test, index_Xtest):
     
    predictions_proba = {}
    models = {}

    for section in sections:

        aux = {}
        aux_models = {}
        
        X_test = dataset[section][index_Xtest]
        
        for name_model in name_models:

            if (name_model != 'mlp') and (name_model != 'mlp_embed') and (name_model != 'mv_mlp_bert'):
                model = joblib.load('{}/test_{}/{}_{}.pkl'.format(path_to_save, num_test, name_model, section))
            else :
                model = load_keras_model(path_to_save, name_model, section, num_test)

            y_pred = model.predict(X_test)

            aux[name_model] = y_pred
            aux_models[name_model] = model

        predictions_proba[section]= aux
        models[section] = aux_models
        
    return predictions_proba, models