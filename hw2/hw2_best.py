import os
import pandas as pd
import numpy as np
import re
import math
import pickle
import matplotlib.pyplot as plt
import sys
from sklearn.svm import SVC

def normalization(X_train, X_test):
    xtrain, xtest = X_train.copy(), X_test.copy()
    age_train = (xtrain['age'] - np.mean(xtrain['age']))/np.std(xtrain['age'])
    fnlwgt_train = (xtrain['fnlwgt'] - np.mean(xtrain['fnlwgt']))/np.std(xtrain['fnlwgt'])
    capital_gain_train = (xtrain['capital_gain'] - np.mean(xtrain['capital_gain']))/np.std(xtrain['capital_gain'])
    capital_loss_train = (xtrain['capital_loss'] - np.mean(xtrain['capital_loss']))/np.std(xtrain['capital_loss'])
    hours_per_week_train = (xtrain['hours_per_week'] - np.mean(xtrain['hours_per_week']))/np.std(xtrain['hours_per_week'])
    
    age_test = (xtest['age'] - np.mean(xtrain['age']))/np.std(xtrain['age'])
    fnlwgt_test = (xtest['fnlwgt'] - np.mean(xtrain['fnlwgt']))/np.std(xtrain['fnlwgt'])
    capital_gain_test = (xtest['capital_gain'] - np.mean(xtrain['capital_gain']))/np.std(xtrain['capital_gain'])
    capital_loss_test = (xtest['capital_loss'] - np.mean(xtrain['capital_loss']))/np.std(xtrain['capital_loss'])
    hours_per_week_test = (xtest['hours_per_week'] - np.mean(xtrain['hours_per_week']))/np.std(xtrain['hours_per_week'])
    
    xtrain['age'] = age_train
    xtrain['fnlwgt'] = fnlwgt_train
    xtrain['capital_gain'] = capital_gain_train
    xtrain['capital_loss'] = capital_loss_train
    xtrain['hours_per_week'] = hours_per_week_train
    
    xtest['age'] = age_test
    xtest['fnlwgt'] = fnlwgt_test
    xtest['capital_gain'] = capital_gain_test
    xtest['capital_loss'] = capital_loss_test
    xtest['hours_per_week'] = hours_per_week_test
    
    return xtrain, xtest


if __name__=="__main__":
    X_train = pd.read_csv(sys.argv[3])
    Y_train = pd.read_csv(sys.argv[4], header=None)
    X_test = pd.read_csv(sys.argv[5])

    X_train_norm, X_test_norm = normalization(X_train, X_test)

    # clf = SVC(gamma=0.2)
    # clf.fit(X_train_norm, Y_train)

    # with open('./HW2_BEST.pickle', 'wb') as file:
    #     pickle.dump(clf, file)

    with open('./HW2_BEST.pickle', 'rb') as file:
        clf = pickle.load(file)

    Y_test = clf.predict(X_test_norm)

    _id = [str(i) for i in range(1,(Y_test.shape[0]+1))]
    data = {'id':_id, 'label': Y_test}
    output = pd.DataFrame(data)
    output.to_csv(path_or_buf=sys.argv[6], index=False)

