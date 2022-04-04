import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.preprocessing import StandardScaler


def cluster_analisys(df_embed, clustering, normalize=True, verbose=False):

  data = {"sentence": [], 'dis_centroid': []}
  scaler = StandardScaler()
  
  for label in np.unique(clustering.labels_):

    if verbose:
        print(label)
      
    indexes = [i for i, x in enumerate(clustering.labels_) if x == label]
    cluster = df_embed.iloc[indexes].reset_index(drop=True)
    centroid = np.asarray(cluster.mean(axis=0))
    if label == -1:
        d = [0]*len(indexes)
    else:
        d = [1 - spatial.distance.cosine(centroid, row) for index, row in cluster.iterrows()]

    if normalize:
        scale = scaler.fit_transform(np.asarray(d).reshape(-1, 1))
        for i in scale:
            data['dis_centroid'].append(float(i))
    else:
        for i in d:
            data['dis_centroid'].append(float(i))

        for i in indexes:
            data['sentence'].append(i)

  cluster_df = pd.DataFrame(data)
  cluster_df = cluster_df.fillna(0)
  cluster_df = cluster_df.sort_values(by='sentence')
  
  return cluster_df