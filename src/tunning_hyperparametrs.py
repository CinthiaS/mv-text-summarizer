from sklearn.model_selection import RandomizedSearchCV
import pickle
import joblib

def randomized_search (parameters, model, X_train, y_train, scoring='precision', n_iter=100, random_state=42):
    
    rsearch = RandomizedSearchCV(
        estimator=model, param_distributions=parameters,
        n_iter=n_iter, scoring=scoring)

    rsearch.fit(X_train, y_train.ravel())
    
    return rsearch

def pipeline_tunning(dataset, models, sections, all_parameters, path_to_save):

    for section in sections:
        
        for name_model in models.keys():
            
            parameters = all_parameters[section][name_model]
            model = models[name_model]

            X = dataset[section][0]
            y = dataset[section][2]

            search = randomized_search (parameters, model, X, y)
            joblib.dump(search, '{}/search_{}_{}.pkl'.format(path_to_save, name_model, section))
