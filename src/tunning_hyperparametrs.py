from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score

def randomized_search (parameters, model, X_train, y_train, n_iter=100, random_state=42):
    
    rsearch = RandomizedSearchCV(
        estimator=model, param_distributions=parameters,
        n_iter=n_iter, random_state=random_state,
        scoring='precision')
    rsearch.fit(X_train, y_train.ravel())
    
    return rsearch

def get_hiperparametrs_svm(X_train, y_train, parameters):

    svmsearch = randomized_search (parameters, SVC(class_weight=parameters['class_weight']), X_train, y_train)

    return svmsearch

def get_hiperparametrs_knn(X_train, y_train, parameters):

    knnsearch = randomized_search (parameters, KNeighborsClassifier(), X_train, y_train)

    return knnsearch

def get_hiperparametrs_rf(X_train, y_train, parameters):
  
    rfsearch = randomized_search (parameters, RandomForestClassifier(class_weight=parameters['class_weight']), X_train, y_train)

    return rfsearch

def get_hiperparametrs_gb(X_train, y_train, parameters):
    
    gbsearch = randomized_search (parameters, GradientBoostingClassifier(), X_train, y_train)

    return gbsearch