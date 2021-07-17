
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

def standart_norm (X):

    scaler = StandardScaler()
    normalized = scaler.fit_transform(X)

    return normalized

def balancing_data (X, y, sampling_strategy='majority'):

  undersample = RandomUnderSampler(sampling_strategy=sampling_strategy)
  X, y = undersample.fit_resample(X, y)

  return X, y