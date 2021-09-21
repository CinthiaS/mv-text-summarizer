from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import tunning_hyperparametrs as th

def get_hiperparametrs_gb(X_train, y_train):

  parameters = {'n_estimators': [50, 100, 200],
               'min_samples_leaf':[5, 10, 20],
               'min_samples_split':[10, 20, 40],
               'max_depth':[5, 7, 20],
              }

  gbsearch = th.randomized_search (parameters, GradientBoostingRegressor(), X_train, y_train)

  return gbsearch

def pipeline_gb(dataset, X, y, X_train, y_train, X_test, y_test, train=False):

    if train:
        gbsearch = get_hiperparametrs_gb(X, y)
        pickle.dump(gbsearch, open('gbsearch_{}'.format(dataset), 'wb'))

        gb = GradientBoostingRegressor(
        n_estimators=gbsearch.best_estimator_.n_estimators, 
        min_samples_leaf=gbsearch.best_estimator_.min_samples_leaf, 
        min_samples_split=gbsearch.best_estimator_.min_samples_split, 
        max_depth=gbsearch.best_estimator_.max_depth)
        gb.fit(X_train, y_train)
        pickle.dump(gb, open('gb_{}'.format(dataset), 'wb'))

    infile = open('gbsearch_{}'.format(dataset),'rb')
    gbsearch = pickle.load(infile)
    infile.close()

    print("\n Gradient Boost Hiperparâmetros")
    print("Num estimators: {}".format(gbsearch.best_estimator_.n_estimators))
    print("Min samples leaf: {}".format(gbsearch.best_estimator_.min_samples_leaf))
    print("Min samples splot: {}".format(gbsearch.best_estimator_.min_samples_split))
    print("Max depth: {}".format(gbsearch.best_estimator_.max_depth))
    print("Best Score: {}".format(gbsearch.best_score_))

    infile = open('gb_{}'.format(dataset),'rb')
    gb = pickle.load(infile)
    infile.close()

    y_pred = gb.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return mse, mae