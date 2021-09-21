from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import tunning_hyperparametrs as th

def get_hiperparametrs_rf(X_train, y_train):
  
  parameters = {'n_estimators': [50,  100, 200],
              'min_samples_leaf':[5, 10, 20],
              'min_samples_split':[10, 20, 40],
              'max_depth':[ 5, 7, 20]
              }

  rfsearch = th.randomized_search (parameters, RandomForestRegressor(), X_train, y_train)

  return rfsearch

def pipeline_rf(dataset, X, y, X_train, y_train, X_test, y_test, train=False):

    if train == True:
        rfsearch = get_hiperparametrs_rf(X, y)
        pickle.dump(rfsearch, open('rfsearch_{}'.format(dataset), 'wb'))

        rf = RandomForestRegressor(
        n_estimators=rfsearch.best_estimator_.n_estimators,
        min_samples_leaf=rfsearch.best_estimator_.min_samples_leaf,
        min_samples_split=rfsearch.best_estimator_.min_samples_split,
        max_depth=rfsearch.best_estimator_.max_depth)
        rf.fit(X_train, y_train)
        pickle.dump(rf, open('rf_{}'.format(dataset), 'wb'))


    infile = open('rfsearch_{}'.format(dataset),'rb')
    rfsearch = pickle.load(infile)
    infile.close()

    print("\n Random Forest Hiperparâmetros")
    print("Num estimators: {}".format(rfsearch.best_estimator_.n_estimators))
    print("Min samples leaf: {}".format(rfsearch.best_estimator_.min_samples_leaf))
    print("Min samples splot: {}".format(rfsearch.best_estimator_.min_samples_split))
    print("Max depth: {}".format(rfsearch.best_estimator_.max_depth))
    print("Best Score: {}".format(rfsearch.best_score_))

    infile = open('rf_{}'.format(dataset),'rb')
    rf = pickle.load(infile)
    infile.close()

    y_pred = rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return mse, mae
