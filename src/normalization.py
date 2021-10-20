
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import pickle

def standart_norm (X):

    scaler = StandardScaler()
    normalized = scaler.fit_transform(X)

    return normalized

def balancing_data (X, y, sampling_strategy='majority'):

    undersample = RandomUnderSampler(sampling_strategy=sampling_strategy)
    X, y = undersample.fit_resample(X, y)

    return X, y

def scale_fit_transform(X_train, section, train=False):
    
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    pickle.dump(scaler, open('scale_{}.pkl'.format(section),'wb'))
    
    X_train = scaler.transform(X_train)
    
    return X_train, scaler

""""
try:
        scaler = pickle.load(open('scale_{}.pkl'.format(section),'rb'))
    except FileNotFoundError:
        pickle.dump(scaler, open('scale_{}.pkl'.format(section),'wb'))
"""