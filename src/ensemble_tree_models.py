from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from src import tunning_hyperparametrs as th
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import pickle
import joblib



def tunning(model, X, y, section, parameters=None):
    
    if parameters == None:
     
        parameters = {'n_estimators': [50,  100, 200],
              'min_samples_leaf':[5, 20, 100],
              'min_samples_split':[10, 40, 200],
              'max_depth':[ 5, 20, 50]
              }

    search = th.randomized_search (parameters, model, X, y)
    joblib.dump(search, '{}_{}.pkl'.format(model, section))

    print("\n Random Forest Hiperparâmetros")
    print("Num estimators: {}".format(search.best_estimator_.n_estimators))
    print("Min samples leaf: {}".format(search.best_estimator_.min_samples_leaf))
    print("Min samples splot: {}".format(search.best_estimator_.min_samples_split))
    print("Max depth: {}".format(search.best_estimator_.max_depth))
    print("Best Score: {}".format(search.best_score_))

    return search

def pipeline(dataset, name_model, section, parameters=None):
    
    if name_model == "gb":
        model = GradientBoostingClassifier()
    elif name_model == "rf":
        model = RandomForestClassifier()
        
    X = dataset[section][0]
    y = dataset[section][2]

    search=tunning(model, X, y, section, parameters=None)
    
    return  search


def gb_classifier(X_train, y_train, section, n_estimators=None, min_samples_leaf=None, min_samples_split=None, max_depth=None):
    
    gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_depth=max_depth)
    
    gb.fit(X_train, y_train)
    
    pickle.dump(gb, open('gb_{}'.format(section), 'wb'))
    
    return gb

def rf_classifier(X_train, y_train, section, n_estimators=None, min_samples_leaf=None, min_samples_split=None, max_depth=None):
    
    rf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_depth=max_depth)
    
    rf.fit(X_train, y_train)
    pickle.dump(rf, open('rf_{}'.format(section), 'wb'))
    
    return rf

def ab_classifier(X_train, y_train, section, n_estimators=100):

    ab = AdaBoostClassifier(n_estimators=100)
    ab.fit(X_train, y_train)
    pickle.dump(ab, open('ab_{}'.format(section), 'wb'))
    
    return ab

def knn_classifier(X_train, y_train, section, n_neighbors=5):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    pickle.dump(knn, open('knn_{}'.format(section), 'wb'))
    
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