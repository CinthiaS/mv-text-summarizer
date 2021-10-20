import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from collections import Counter

import seaborn as sns
sns.set_theme(style="whitegrid")

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