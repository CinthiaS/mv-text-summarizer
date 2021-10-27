import pandas as pd
import numpy as np
import pickle

from src import utils_classification as utils
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report

from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt

def evaluate_classification(X_test, y_test, model, name_section, name_model, columns_name, verbose=False):

    results = {}

    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.insert(loc=0, column='model', value=[name_model]*5)
    
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


def roc_curve(sections, dataset):
    
    for section in sections:
        
    
        X_test =  dataset[section][1]
        y_test =  dataset[section][3]
    
        knn, gbc, rfc, abc = models.get(section)

        rf = plot_roc_curve(rfc, X_test, y_test)
        ab = plot_roc_curve(abc, X_test, y_test, ax=rf.ax_)
        gb = plot_roc_curve(gbc, X_test, y_test, ax=ab.ax_)
        knn = plot_roc_curve(knn, X_test, y_test, ax=gb.ax_)

        knn.figure_.suptitle("ROC curve comparison - {}".format(section))
        plt.show()
        
def convert_pred(y_pred):
    
    y_pred[y_pred == 1] = 1
    y_pred[y_pred == 0] = -1
    
    return y_pred
    
def matthews(sections, dataset, predictions):

    result = {}
    
    for section in sections:
        
        aux = {}
        y_test = dataset[section][3].copy()
        
        aux['knn'] = matthews_corrcoef(y_test, predictions['knn_{}'.format(section)])
        aux['rf'] = matthews_corrcoef(y_test,predictions['rf_{}'.format(section)])
        aux['gb'] = matthews_corrcoef(y_test, predictions['gb_{}'.format(section)])
        aux['ab'] = matthews_corrcoef(y_test, predictions['ab_{}'.format(section)])
        
        result[section] = aux 
        
    df = pd.DataFrame(result)
        
    return df