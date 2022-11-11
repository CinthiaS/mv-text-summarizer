from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import pickle
import joblib
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, concatenate, Dropout, Input

def gb_classifier(
    X_train, y_train, section,parameters, num_test=1, path_to_save='models'):
    
    print('gb')
    
    gb = GradientBoostingClassifier(
            n_estimators=parameters['gb'][section]['n_estimators'], 
                min_samples_leaf=parameters['gb'][section]['min_samples_leaf'],
                min_samples_split=parameters['gb'][section]['min_samples_split'],
                max_depth=parameters['gb'][section]['max_depth'])
    
    gb.fit(X_train, y_train)
    
    pickle.dump(gb, open('{}/test_{}/gb_{}.pkl'.format(path_to_save, num_test, section), 'wb'))
    
    return gb

def cb_classifier(X_train, y_train, section,parameters, num_test=1, path_to_save='models'):
    
    print('cb')
    
    cb = CatBoostClassifier(
            iterations= parameters['cb'][section]['iterations'],
            learning_rate= parameters['cb'][section]['learning_rate'],
            depth= parameters['cb'][section]['depth'],
            min_data_in_leaf=parameters['cb'][section]['min_data_in_leaf'])
    
    cb.fit(X_train, y_train, verbose=False)
    
    pickle.dump(cb, open('{}/test_{}/cb_{}.pkl'.format(path_to_save, num_test, section), 'wb'))
    
    return cb

def rf_classifier(
        X_train, y_train, section, parameters, n_jobs=-1, num_test=1, path_to_save='models'):
    
    print('rf')
    
    rf = RandomForestClassifier(
            n_estimators=parameters['rf'][section]['n_estimators'], 
            min_samples_leaf=parameters['rf'][section]['min_samples_leaf'],
            min_samples_split=parameters['rf'][section]['min_samples_split'],
            max_depth=parameters['rf'][section]['max_depth'],
            n_jobs=n_jobs)
    
    rf.fit(X_train, y_train)
    
    pickle.dump(rf, open('{}/test_{}/rf_{}.pkl'.format(path_to_save, num_test, section), 'wb'))
    
    return rf


def ab_classifier(X_train, y_train, section, parameters, num_test=1, path_to_save='models'):
    
    print('ab')

    ab = AdaBoostClassifier(
        n_estimators=parameters['ab'][section]['n_estimators'])
    ab.fit(X_train, y_train)
    
    pickle.dump(ab, open('{}/test_{}/ab_{}.pkl'.format(path_to_save, num_test, section), 'wb'))
    
    return ab

def svm_classifier(X_train, y_train, section, parameters, num_test=1, path_to_save='models'):

    svm = SVC(
        kernel=parameters['svm'][section]['kernel'], 
        degree=parameters['svm'][section]['degree'],
        class_weight=parameters['svm'][section]['class_weight'])
    
    svm.fit(X_train, y_train)
    
    pickle.dump(svm, open('{}/test_{}/svm_{}.pkl'.format(path_to_save, num_test, section), 'wb'))
    
    return svm

def mlp_classifier(
    X_train, y_train, section, test_size=0.2, batch_size=4,
    learning_rate=0.0001, epochs=10,  num_test=1, path_to_save='models'):
    
    print('mlp')

    sequence_input = Input(shape=(X_train.shape[1],), dtype='int32')

    perceptron_1 = Dense(256, activation='relu')(sequence_input)
    dropout1 = Dropout(.2)(perceptron_1)
    perceptron_2 = Dense(256, activation='relu')(dropout1)
    dropout2 = Dropout(.2)(perceptron_2)
    perceptron_3 = Dense(512, activation='relu')(dropout2)
    dropout3 = Dropout(.3)(perceptron_3)
    perceptron_7 = Dense(512, activation='relu')(dropout3)
    dropout7 = Dropout(.3)(perceptron_7)
    perceptron_8 = Dense(256, activation='relu')(dropout7)
    dropout8 = Dropout(.3)(perceptron_8)
    perceptron_9 = Dense(256, activation='relu')(dropout8)
    dropout9 = Dropout(.2)(perceptron_9)

    preds = Dense(2, activation='sigmoid')(dropout9)

    model = Model(inputs=sequence_input, outputs=preds)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate), metrics=['accuracy'])

    one_hot_label = to_categorical(y_train)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, one_hot_label, stratify=one_hot_label, shuffle=True, test_size=test_size)

    history= model.fit(
        x=X_train, y=y_train,
        epochs=epochs, validation_data=(X_valid, y_valid),
         shuffle=True, batch_size=batch_size, verbose=0)
    
    model_json = model.to_json()
    with open('{}/test_{}/mlp_{}.json'.format(path_to_save, num_test, section), "w") as json_file:
        json_file.write(model_json)
    model.save_weights('{}/test_{}/mlp_{}.h5'.format(path_to_save, num_test, section))
  
    
    return model

def knn_classifier(X_train, y_train, section, parameters, n_jobs=-1, num_test=1, path_to_save='models'):
    
    print('knn')

    knn = KNeighborsClassifier(
        n_neighbors=parameters['knn'][section]['n_neighbors'],
        n_jobs=n_jobs)
    knn.fit(X_train, y_train)
    
    pickle.dump(knn, open('{}/test_{}/knn_{}.pkl'.format(path_to_save, num_test, section), 'wb'))
    
    return knn

def pipeline_classifiers(
    dataset, parameters, sections, name_models, index_Xtrain, index_ytrain, num_test=1, path_to_save='models'):
    
    models = {}
    
    for section in sections:
        
        trained = {}
        X_train, y_train = dataset[section][index_Xtrain], dataset[section][index_ytrain]
        
        for name_model in name_models:
    
            if name_model == 'knn':
                trained['knn'] = knn_classifier(
                    X_train, y_train, section, parameters, num_test=num_test, path_to_save=path_to_save)
            elif name_model == 'gb':
                trained['gb'] = gb_classifier(
                    X_train, y_train, section, parameters, num_test=num_test, path_to_save=path_to_save)   
            elif name_model == 'rf':   
                trained['rf'] = rf_classifier(
                    X_train, y_train, section, parameters, num_test=num_test, path_to_save=path_to_save)
            elif name_model == 'ab':    
                trained['ab'] = ab_classifier(
                    X_train, y_train, section, parameters, num_test=num_test, path_to_save=path_to_save)
            elif name_model == 'mlp':
                trained['mlp'] = mlp_classifier(
                    X_train, y_train, section, num_test=num_test, path_to_save=path_to_save)
            elif name_model == 'svm':
                trained['svm'] = svm_classifier(
                    X_train, y_train, section, parameters, num_test=num_test, path_to_save=path_to_save)
            elif name_model == 'cb':
                trained['cb'] = cb_classifier(
                    X_train, y_train, section, parameters, num_test=num_test, path_to_save=path_to_save)
        
        models[section]=trained
   
    return models

