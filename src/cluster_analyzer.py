import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.preprocessing import StandardScaler

def cluster_analisys(df_embed, clustering, normalize=True, verbose=False):

  data = {"sentence": [], 'dis_centroid': []}
  scaler = StandardScaler()
  

  d = [0 for i in clustering.labels_]
    
  for label in np.unique(clustering.labels_):
    
    if label != -1: 
    
      if verbose:
        print(label)
      
      indexes = [i for i, x in enumerate(clustering.labels_) if x == label]
      cluster = df_embed.iloc[indexes]
      centroid = np.asarray(cluster.mean(axis=0))
      
      for i in indexes:
          d[i] = 1 - spatial.distance.cosine(centroid, cluster.loc[i])
            

      if normalize:
        scale = scaler.fit_transform(np.asarray(d).reshape(-1, 1))
        return scale
  return d