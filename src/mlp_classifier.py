from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import model_from_json

from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)
lr_sch = LearningRateScheduler(scheduler)


def save_nn(model, name):
    
    model_json = model.to_json()

    with open("{}.json".format(name), "w") as json_file:
        json_file.write(model_json)

    model.save_weights("{}.h5".format(name))
    
def load_model(dataset):
    
    json_file = open('nn_model_{}.json'.format(dataset), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('nn_model_{}.h5'.format(dataset))
    
    return model

def predict_nn(model, X_test):
    
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    return y_pred

def create_report(y_true, y_pred):
    
    report = classification_report(y_true, y_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()

    return report

def eval_nn(dataset, sections):
    
    results = {}
    predictions = {}
    
    for section in sections:
        
        if section == 'introduction':
            model = load_model('introduction')
        elif section == 'materials':
            model = load_model('materials')
        elif section == 'conclusion':
            model = load_model('conclusion')
        elif section == 'concat': 
            model = load_model('concat')
    
        X_test =  dataset[section][1]
        y_test =  dataset[section][3]
        
        y_pred = predict_nn(model, X_test)
        report = create_report(y_test, y_pred)

        results[section] = report
        predictions[section] = report
        
    return predictions, results
    
    
def train_nn(model, X, y, test_size=0.2, epochs=1000, batch_size=64, verbose=0):    

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=test_size)

    history = model.fit(
        np.array(X_train), np.array(y_train), validation_data=(X_valid,y_valid), epochs=epochs,
         batch_size=batch_size, verbose=verbose)
    
    return model, history

def main_train_nn(dataset, sections, parameters, train=True, verbose=False):

    models = {}

    for section in sections:
        
        if verbose:
            print("\n{}".format(section))

        if section == 'concat':
            model = nn_concat()
        elif section == 'introduction':
            model = nn_intro()
        elif section == 'materials':
            model = nn_mat()
        elif section == 'conclusion':
            model = nn_conc()

        X_train = dataset[section][0]
        y_train = dataset[section][2]
        one_hot_label = to_categorical(y_train)

        test_size, epochs, batch_size =parameters.get(section)

        model, history = train_nn(
            model, X_train, one_hot_label, test_size=test_size, epochs=epochs,
            batch_size=batch_size, verbose=verbose)

        save_nn(model, "nn_model_{}".format(section))
        
        models[section] = model
            
    return models


def cross_validation_nn(model, X, y, epochs=500, batch_size=64, n_splits=5, verbose=0):

    model = KerasClassifier(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=verbose)
    kfold = StratifiedKFold(n_splits=n_splits)
    results = cross_val_score(model, np.array(X), np.array(y), cv=kfold)
    
    if verbose:
        print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    return result

def nn_concat():

    model = Sequential()
    model.add(Dense(128, input_dim=11, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(
            learning_rate=0.001), metrics=['accuracy', keras.metrics.AUC()])
    
    return model

def nn_intro():
    
    model = Sequential()
    model.add(Dense(128, input_dim=11, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(
            learning_rate=0.001), metrics=['accuracy', keras.metrics.AUC()])
    
    return model

def nn_mat():
    
    model = Sequential()
    model.add(Dense(128, input_dim=11, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(
        learning_rate=0.001), metrics=['accuracy', keras.metrics.AUC()])
    
    return model

def nn_conc():
    
    model = Sequential()
    model.add(Dense(128, input_dim=11, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(
            learning_rate=0.001), metrics=['accuracy', keras.metrics.AUC()])
    
    return model