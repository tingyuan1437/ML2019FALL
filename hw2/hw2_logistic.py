import os
import pandas as pd
import numpy as np
import re
import math
import pickle
import matplotlib.pyplot as plt
import sys

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

class LogisticRegression():
    def __init__(self, weight=None):
        self.w = weight
        self.accuracy_history = []
        self.cee_history = []
    
    def fit(self, X_train, Y_train, learning_rate, iteration, add_intercept=False):
        if add_intercept:
            b = np.ones(X_train.shape[0]).reshape(-1,1)
            X_train = np.hstack((b, X_train))
        
        self.w = np.full(X_train[0].shape, 1.0).reshape(-1, 1)
        for i in range(iteration):
            z = np.matmul(X_train, self.w)
            f_x = np.apply_along_axis(self.__sigmoid, 1, z).reshape(-1,1)
            
            accuracy, cee = self.__get_accuracy_cee(Y_train, f_x)
            self.accuracy_history.append(accuracy)
            self.cee_history.append(cee)
            
            error = Y_train - f_x
            grad = np.matmul(X_train.T, error)
            self.w += learning_rate*grad

            print("Iteration: {}, Accuracy: {}, Loss: {}".format(i, accuracy, cee), end = '\r')
    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __classify(self, x):
        return 1 if x > 0.5 else 0

    def __get_accuracy_cee(self, y, f_x):
        y, f_x = y.reshape(-1).tolist(), f_x.reshape(-1).tolist()
        cnt, cee = 0, 0
        epsilon = 1e-7
        for i in range(len(y)):
            cee -= (y[i]*np.log(f_x[i] + epsilon) + (1 - y[i])*np.log(1 - f_x[i] + epsilon))
            if self.__classify(y[i]) == self.__classify(f_x[i]):
                cnt += 1
        return cnt/len(y), cee
    
    def predict(self, X_test, add_intercept=False):
        z = np.matmul(X_test, self.w)
        f_x = np.apply_along_axis(self.__sigmoid, 1, z).reshape(-1,1)
        Y_test = np.apply_along_axis(self.__classify, 1, z).reshape(-1,1)
        return Y_test

if __name__ == '__main__':
    X_train = pd.read_csv(sys.argv[3])
    Y_train = pd.read_csv(sys.argv[4], header=None)
    X_test = pd.read_csv(sys.argv[5])

    X_train_norm, X_test_norm = normalization(X_train, X_test)

    # LR = LogisticRegression()
    # LR.fit(X_train_norm.values, Y_train.values, 0.00001, 2500)

    # file = open('HW2_Logistic.pickle', 'wb')
    # pickle.dump(LR.w, file)
    # file.close()

    with open('./HW2_Logistic.pickle', 'rb') as file:
        weight = pickle.load(file)

    LR = LogisticRegression(weight)
    Y_test = LR.predict(X_test_norm.values)

    _id = [str(i) for i in range(1,(Y_test.shape[0]+1))]
    data = {'id':_id, 'label': Y_test.flatten().tolist()}
    output = pd.DataFrame(data)
    output.to_csv(path_or_buf=sys.argv[6], index=False)