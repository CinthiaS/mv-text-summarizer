import smogn

def load_smong_features(section, columns_name):

    features_smogn = pd.read_csv("smogn_{}.csv".format(section))
    X_train = features_smogn[columns_name]
    y_train = features_smogn['rouge_1']
    
    return X_train, y_train

def balanced_data(X, y, section, metric='rouge_1'):

    X[metric] = y[metric]*100

    X.to_csv("X_{}.csv".format(section), index=False)

    df = pd.read_csv("X_{}.csv".format(section))

    features_smogn = smogn.smoter(

        data = df, 
        y = metric
    )
    
    features_smogn.to_csv("smogn_{}.csv".format(section), index=False)
    
    return features_smogn