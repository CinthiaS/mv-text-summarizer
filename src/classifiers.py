import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src import normalization


def evaluate_model(X_test, y_test, model):

  y_pred = model.predict(X_test)
  scores = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()

  return scores, y_pred


def fit_svm(X_train, y_train, svmsearch):

  svm = SVC(kernel=svmsearch.best_estimator_.kernel,
        gamma=svmsearch.best_estimator_.gamma,
        degree=svmsearch.best_estimator_.degree, probability=True)

  svm.fit(X_train, y_train)

  return svm

def fit_knn(X_train, y_train, knnsearch):

  knn = KNeighborsClassifier(n_neighbors=knnsearch.best_estimator_.n_neighbors,
        weights=knnsearch.best_estimator_.weights,
        algorithm=knnsearch.best_estimator_.algorithm)

  knn.fit(X_train, y_train)

  return knn

def fit_rf(X_train, y_train, rfsearch):

  rf = RandomForestClassifier(n_estimators=rfsearch.best_estimator_.n_estimators,
      criterion=rfsearch.best_estimator_.criterion,
          max_features=rfsearch.best_estimator_.max_features,
          min_samples_leaf=rfsearch.best_estimator_.min_samples_leaf,
          min_samples_split=rfsearch.best_estimator_.min_samples_split,
          max_depth=rfsearch.best_estimator_.max_depth)

  rf.fit(X_train, y_train)

  return rf

def fit_gb(X_train, y_train, gbsearch):

  gb = GradientBoostingClassifier(n_estimators=gbsearch.best_estimator_.n_estimators,
      criterion=gbsearch.best_estimator_.criterion,
          max_features=gbsearch.best_estimator_.max_features,
          min_samples_leaf=gbsearch.best_estimator_.min_samples_leaf,
          min_samples_split=gbsearch.best_estimator_.min_samples_split,
          max_depth=gbsearch.best_estimator_.max_depth)

  gb.fit(X_train, y_train)

  return gb

def data_classification(X, y, test_size=0.3, sampling_strategy='majority', random_state=None):

  X = normalization.standart_norm(X)
  
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y)

  X_train, y_train = normalization.balancing_data (X_train, y_train, sampling_strategy='majority')

  return X_train, X_test, y_train, y_test

