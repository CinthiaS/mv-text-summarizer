import pandas as pd
import numpy as np
import pickle

from src import utils_classification as utils
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report

from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt

def evaluate_classification(X_test, y_test, model, name_section, name_model, verbose=False):

    results = {}

    if name_model != "mlp":
        y_pred = model.predict(X_test)
    
    y_proba = model.predict_proba(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.insert(loc=0, column='model', value=[name_model]*5)
    
    if verbose:
        print("Report {}:\n{}\n".format(name_section, report))

    return y_pred, y_proba, report

def create_reports(models, dataset, name_models, index_Xtest, index_ytest, verbose=False):
    
    results = {}
    predictions = {}
    predictions_proba = {}
    
    for i in models.keys():

        model = models.get(i)
        X = dataset.get(i)[index_Xtest]
        y = dataset.get(i)[index_ytest]
        
        aux_predict = {}
        aux_results = {}
        aux_predict_proba = {}
        
        for name_model in name_models:

            aux_predict[name_model], aux_predict_proba[name_model], aux_results[name_model] = evaluate_classification(
                X, y, model[name_model], name_section=i, name_model=name_model, verbose=verbose)
            
        results[i] = aux_results
        predictions[i] = aux_predict
        predictions_proba[i] = aux_predict_proba

    return predictions, predictions_proba, results


def roc_curve(sections, dataset, models):
    
    for section in sections:
        
    
        X_test =  dataset[section][1]
        y_test =  dataset[section][3]
    
        model = models.get(section)

        rf = plot_roc_curve(model['rf'], X_test, y_test)
        ab = plot_roc_curve(model['ab'], X_test, y_test, ax=rf.ax_)
        gb = plot_roc_curve(model['gb'], X_test, y_test, ax=ab.ax_)
        knn = plot_roc_curve(model['knn'], X_test, y_test, ax=gb.ax_)
        mlp = plot_roc_curve(model['mlp'], X_test, y_test, ax=knn.ax_)

        mlp.figure_.suptitle("ROC curve comparison - {}".format(section))
        plt.show()
        
def convert_pred(y_pred):
    
    y_pred[y_pred == 1] = 1
    y_pred[y_pred == 0] = -1
    
    return y_pred
    
def matthews(sections, dataset, predictions, name_models, index_ytest):

    result = {}
    
    for section in sections:
        
        aux = {}
        y_test = dataset[section][index_ytest].copy()
        
        for name_model in name_models:
        
            aux[name_model] = matthews_corrcoef(y_test, predictions[section][name_model])
            
        result[section]=aux
        
    df = pd.DataFrame(result)

    return df


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
    
