from src import tunning_hyperparametrs as th
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import pickle
import joblib

def tunning(model, X, y, section, path_to_save='models', parameters=None):

    search = th.randomized_search (parameters, model, X, y)
    joblib.dump(search, '{}/{}_{}.pkl'.format(path_to_save, model, section))

    print("Num estimators: {}".format(search.best_estimator_.n_estimators))
    print("Min samples leaf: {}".format(search.best_estimator_.min_samples_leaf))
    print("Min samples splot: {}".format(search.best_estimator_.min_samples_split))
    print("Max depth: {}".format(search.best_estimator_.max_depth))
    print("Best Score: {}".format(search.best_score_))

    return search

def pipeline(dataset, name_model, section, n_jobs=-1, parameters=None):
    
    if name_model == "gb":
        model = GradientBoostingClassifier()
    elif name_model == "rf":
        model = RandomForestClassifier(n_jobs=n_jobs)
    elif name_model == 'ab':
        model = AdaBoostClassifier()
    elif name_model == 'knn':
        model = KNeighborsClassifier(n_jobs=n_jobs)
        
    X = dataset[section][0]
    y = dataset[section][2]

    search=tunning(model, X, y, section, parameters=parameters)
    
    return  search


def gb_classifier(
    X_train, y_train, section, n_estimators=None, min_samples_leaf=None,
    min_samples_split=None, max_depth=None, path_to_save='models'):
    
    gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_depth=max_depth, n_jobs=n_jobs)
    
    gb.fit(X_train, y_train)
    
    pickle.dump(gb, open('{}/gb_{}'.format(path_to_save, section), 'wb'))
    
    return gb

def rf_classifier(
        X_train, y_train, section, n_estimators=None, min_samples_leaf=None,
        min_samples_split=None, max_depth=None, n_jobs=-1, path_to_save='models'):
    
    rf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_depth=max_depth, n_jobs=n_jobs,)
    
    rf.fit(X_train, y_train)
    pickle.dump(rf, open('{}/rf_{}'.format(path_to_save, section), 'wb'))
    
    return rf

def ab_classifier(X_train, y_train, section, n_estimators=100, path_to_save='models'):

    ab = AdaBoostClassifier(n_estimators=n_estimators)
    ab.fit(X_train, y_train)
    pickle.dump(ab, open('{}/ab_{}'.format(path_to_save, section), 'wb'))
    
    return ab

def svm_classifier(X_train, y_train, section, n_estimators=100, path_to_save='models'):

    ab = SVC()
    ab.fit(X_train, y_train)
    pickle.dump(ab, open('{}/svm_{}'.format(path_to_save, section), 'wb'))
    
    return ab

def knn_classifier(X_train, y_train, section, n_neighbors=5, n_jobs=-1, path_to_save='models'):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    knn.fit(X_train, y_train)
    pickle.dump(knn, open('{}/knn_{}'.format(path_to_save, section), 'wb'))
    
    return knn

def pipeline_classifiers(X_train, y_train, section, name_models=['knn', 'gb', 'rf', 'ab']):
    
    trained = []
    for i in name_models:
        
        if i == 'knn':
            knn = knn_classifier(
                X_train, y_train, section, n_neighbors=5)
            trained.append(knn)
        elif i == 'gb':
            gb = gb_classifier(
                X_train, y_train, section, n_estimators=200, min_samples_leaf=20,
                min_samples_split=40, max_depth=20)
            trained.append(gb)
        elif i == 'rf':
            rf = rf_classifier(
                X_train, y_train, section, n_estimators=100, min_samples_leaf=10,
                min_samples_split=20, max_depth=20)
            trained.append(rf)
        elif i == 'ab':
            ab = ab_classifier(X_train, y_train, section, n_estimators=100)
            trained.append(ab)
    
    return trained

def create_models(dataset, sections, name_models):

    models = {}

    for section in sections:
    
        trained = pipeline_classifiers(dataset[section][0], dataset[section][2], section, name_models)
        models[section] = trained
        
    return models