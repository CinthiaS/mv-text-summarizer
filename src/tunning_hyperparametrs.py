from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


def randomized_search (parameters, model, X_train, y_train, n_iter=100, random_state=42):
    
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=n_iter, random_state=random_state)
    rsearch.fit(X_train, y_train.ravel())
    
    return rsearch

def get_hiperparametrs_svm(X_train, y_train, parameters):

  parameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
              'gamma': ('auto', 'scale'),
              'degree':[1, 2, 3, 4, 5, 20],
              }
  svmsearch = randomized_search (parameters, SVC(), X_train, y_train)

  return svmsearch

def get_hiperparametrs_knn(X_train, y_train, parameters):

  parameters = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'weights': ('uniform', 'distance'),
              'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')
              }

  knnsearch = randomized_search (parameters, KNeighborsClassifier(), X_train, y_train)

  return knnsearch

def get_hiperparametrs_rf(X_train, y_train, parameters):
  
  parameters = {'n_estimators': [10, 20, 40, 80, 100, 150, 200],
              'max_features':('sqrt', 'log2', 'auto'),
              'criterion': ('gini', 'entropy'),
              'min_samples_leaf':[10, 15, 20],
              'min_samples_split':[20, 30, 40],
              'max_depth':[1, 2, 5, 7],
              }

  rfsearch = randomized_search (parameters, RandomForestClassifier(), X_train, y_train)

  return rfsearch

def get_hiperparametrs_gb(X_train, y_train, parameters):

  parameters = {'n_estimators': [10, 20, 40, 80, 100, 150, 200],
              'max_features':('sqrt', 'log2', 'auto'),
              'min_samples_leaf':[10, 15, 20],
              'min_samples_split':[20, 30, 40],
              'max_depth':[1, 2, 5, 7],
              }

  gbsearch = randomized_search (parameters, GradientBoostingClassifier(), X_train, y_train)

  return gbsearch