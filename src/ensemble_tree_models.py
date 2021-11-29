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


def gb_classifier(
    X_train, y_train, section,parameters, path_to_save='models'):
    
    gb = GradientBoostingClassifier(
            n_estimators=parameters['gb'][section]['n_estimators'], 
                min_samples_leaf=parameters['gb'][section]['min_samples_leaf'],
                min_samples_split=parameters['gb'][section]['min_samples_split'],
                max_depth=parameters['gb'][section]['max_depth'])
    
    gb.fit(X_train, y_train)
    
    pickle.dump(gb, open('{}/gb_{}.pkl'.format(path_to_save, section), 'wb'))
    
    return gb

def cb_classifier(X_train, y_train, section,parameters, path_to_save='models'):
    
    
    cb = CatBoostClassifier(
            iterations= parameters['cb'][section]['iterations'],
            learning_rate= parameters['cb'][section]['learning_rate'],
            depth= parameters['cb'][section]['depth'],
            class_weights=parameters['cb'][section]['class_weights'],
            min_data_in_leaf=parameters['cb'][section]['min_data_in_leaf'])
    
    cb.fit(X_train, y_train, verbose=False)
    
    pickle.dump(cb, open('{}/cb_{}.pkl'.format(path_to_save, section), 'wb'))
    
    return cb

def rf_classifier(
        X_train, y_train, section, parameters, n_jobs=-1, path_to_save='models'):
    
    rf = RandomForestClassifier(
            n_estimators=parameters['rf'][section]['n_estimators'], 
            min_samples_leaf=parameters['rf'][section]['min_samples_leaf'],
            min_samples_split=parameters['rf'][section]['min_samples_split'],
            max_depth=parameters['rf'][section]['max_depth'],
            class_weight=parameters['rf'][section]['class_weight'],
            n_jobs=n_jobs)
    
    rf.fit(X_train, y_train)
    
    pickle.dump(rf, open('{}/rf_{}.pkl'.format(path_to_save, section), 'wb'))
    
    return rf


def ab_classifier(X_train, y_train, section, parameters, path_to_save='models'):

    ab = AdaBoostClassifier(
        n_estimators=parameters['ab'][section]['n_estimators'])
    ab.fit(X_train, y_train)
    
    pickle.dump(ab, open('{}/ab_{}.pkl'.format(path_to_save, section), 'wb'))
    
    return ab

def svm_classifier(X_train, y_train, section, parameters, path_to_save='models'):

    svm = SVC(
        kernel=parameters['svm'][section]['kernel'], 
        degree=parameters['svm'][section]['degree'],
        class_weight=parameters['svm'][section]['class_weight'])
    
    svm.fit(X_train, y_train)
    
    pickle.dump(svm, open('{}/svm_{}.pkl'.format(path_to_save, section), 'wb'))
    
    return ab

def mlp_classifier(X_train, y_train, section, parameters, path_to_save='models'):

    mlp = MLPClassifier(
        max_iter=parameters['mlp'][section]['max_iter'],
        batch_size=parameters['mlp'][section]['batch_size'],
        hidden_layer_sizes=parameters['mlp'][section]['hidden_layer_sizes'])
    
    mlp.fit(X_train, y_train)
    
    pickle.dump(mlp, open('{}/mlp_{}.pkl'.format(path_to_save, section), 'wb'))
    
    return mlp

def knn_classifier(X_train, y_train, section, parameters, n_jobs=-1, path_to_save='models'):

    knn = KNeighborsClassifier(
        n_neighbors=parameters['knn'][section]['n_neighbors'],
        n_jobs=n_jobs)
    knn.fit(X_train, y_train)
    
    pickle.dump(knn, open('{}/knn_{}.pkl'.format(path_to_save, section), 'wb'))
    
    return knn

def pipeline_classifiers(dataset, parameters, sections, name_models, path_to_save='models'):
    
    models = {}
    
    for section in sections:
        
        trained = {}
        X_train, y_train = dataset[section][0], dataset[section][2]
        
        for name_model in name_models:
    
            if name_model == 'knn':
                trained['knn'] = knn_classifier(X_train, y_train, section, parameters, path_to_save=path_to_save)
            elif name_model == 'gb':
                trained['gb'] = gb_classifier(X_train, y_train, section, parameters, path_to_save=path_to_save)   
            elif name_model == 'rf':   
                trained['rf'] = rf_classifier(X_train, y_train, section, parameters, path_to_save=path_to_save)
            elif name_model == 'ab':    
                trained['ab'] = ab_classifier(X_train, y_train, section, parameters, path_to_save=path_to_save)
            elif name_model == 'mlp':
                trained['mlp'] = mlp_classifier(X_train, y_train, section, parameters, path_to_save=path_to_save)
            elif name_model == 'svm':
                trained['svm'] = svm_classifier(X_train, y_train, section, parameters, path_to_save=path_to_save)
            elif name_model == 'cb':
                trained['cb'] = cb_classifier(X_train, y_train, section, parameters, path_to_save=path_to_save)
        
        models[section]=trained
   
    return models

def create_models(dataset, parameters, sections, name_models):
    
    models = {}

    for section in sections:
        
        for name_model in name_models:
            
            models[section] = pipeline_classifiers(
                dataset[section][0], dataset[section][2], parameters, section, name_model)
