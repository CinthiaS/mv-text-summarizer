import pandas as pd
import numpy as np
import pickle

from src import utils_classification as utils
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report

def evaluate_classification(X_test, y_test, model, name_section, name_model, columns_name, verbose=False):

    results = {}

    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.insert(loc=0, column='model', value=[name_model]*5)
    
    mcc = matthews_corrcoef(y_test, y_pred)
    
    if verbose:
        print("Report {}:\n{}\n".format(name_section, report))

    return y_pred, report

def create_reports(models, dataset, columns_name, verbose=False):
    
    results = {}
    predictions = {}
    
    for i in models.keys():

        aux = models.get(i)
        X = dataset.get(i)[1]
        y = dataset.get(i)[3]

        predictions['knn_{}'.format(i)], results['knn_{}'.format(i)] = evaluate_classification(
            X, y, aux[0], name_section=i, name_model='knn_{}'.format(i), columns_name=columns_name, verbose=verbose)
        predictions['ab_{}'.format(i)], results['ab_{}'.format(i)] = evaluate_classification(
            X, y, aux[1], name_section=i, name_model='ab_{}'.format(i), columns_name=columns_name, verbose=verbose)
        predictions['gb_{}'.format(i)], results['gb_{}'.format(i)] = evaluate_classification(
            X, y, aux[2], name_section=i, name_model='gb_{}'.format(i), columns_name=columns_name, verbose=verbose)
        predictions['rf_{}'.format(i)], results['rf_{}'.format(i)] = evaluate_classification(
            X, y, aux[3], name_section=i, name_model='rf_{}'.format(i), columns_name=columns_name, verbose=verbose)

    return predictions, results