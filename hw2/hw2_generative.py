import os
import pandas as pd
import numpy as np
import re
import math
import pickle
import matplotlib.pyplot as plt
import sys

def preprocess(X_train, Y_train):
    tmp = X_train.copy()
    tmp['y'] = Y_train.values
    X_train_1 = tmp.loc[tmp['y'] == 1]
    X_train_0 = tmp.loc[tmp['y'] == 0]
    X_train_1 = X_train_1.drop(['y'], axis=1)
    X_train_0 = X_train_0.drop(['y'], axis=1)
    return X_train_1, X_train_0

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

class ProbabilisticGenerativeClassifier():
    def __init__(self,b1=None,w1=None,b0=None,w0=None):
        self.b1 = b1
        self.w1 = w1
        self.b0 = b0
        self.w0 = w0
    
    def fit(self, X_train_1, X_train_0):
        prior_1 = X_train_1.shape[0]/(X_train_1.shape[0] + X_train_0.shape[0])
        prior_0 = X_train_0.shape[0]/(X_train_1.shape[0] + X_train_0.shape[0])
        X_train_1 = X_train_1.T
        X_train_0 = X_train_0.T
        mu_1 = np.mean(X_train_1, axis=1).reshape(-1,1)
        mu_0 = np.mean(X_train_0, axis=1).reshape(-1,1)
        cov_1 = np.cov(X_train_1)
        cov_0 = np.cov(X_train_0)
        shared_cov = prior_1*cov_1 + prior_0*cov_0
        shared_cov_inv = np.linalg.inv(shared_cov)
        self.w1 = np.matmul(shared_cov_inv, (mu_1-mu_0))
        self.b1 = (-1/2)*np.matmul(np.matmul(mu_1.T, shared_cov_inv), mu_1) + (1/2)*np.matmul(np.matmul(mu_0.T, shared_cov_inv), mu_0) + np.log(prior_1/prior_0)
        self.w0 = np.matmul(shared_cov_inv, (mu_0-mu_1))
        self.b0 = (-1/2)*np.matmul(np.matmul(mu_0.T, shared_cov_inv), mu_0) + (1/2)*np.matmul(np.matmul(mu_1.T, shared_cov_inv), mu_1) + np.log(prior_0/prior_1)
    
    def predict(self, X_test):
        b1_mat = np.full(X_test.shape[0], self.b1)
        b0_mat = np.full(X_test.shape[0], self.b0)
        X_test = X_test.T
        z1 = np.matmul(self.w1.T, X_test) + b1_mat
        z0 = np.matmul(self.w0.T, X_test) + b0_mat
        p1 = np.apply_along_axis(self.__sigmoid, 0, z1)
        p0 = np.apply_along_axis(self.__sigmoid, 0, z0)
        return self.__compare(p1.flatten().tolist(), p0.flatten().tolist())
    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __compare(self, p1, p0):
        ret = []
        for i in range(len(p1)):
            if p1[i] > p0[i]:
                ret.append(1)
            else:
                ret.append(0)
        return np.asarray(ret).reshape(-1,1)

if __name__ == '__main__':
    X_train = pd.read_csv(sys.argv[3])
    Y_train = pd.read_csv(sys.argv[4], header=None)
    X_test = pd.read_csv(sys.argv[5])

    X_train_norm, X_test_norm = normalization(X_train, X_test)
    X_train_1, X_train_0 = preprocess(X_train_norm, Y_train)

    # pgc = ProbabilisticGenerativeClassifier()
    # pgc.fit(X_train_1.values, X_train_0.values)

    # pkl = [pgc.b1, pgc.w1, pgc.b0, pgc.w0]

    # file = open('./HW2_PGC.pickle', 'wb')
    # pickle.dump(pkl, file)
    # file.close()

    with open('./HW2_PGC.pickle', 'rb') as file:
        weight = pickle.load(file)

    pgc = ProbabilisticGenerativeClassifier(weight[0],weight[1],weight[2],weight[3])

    Y_test = pgc.predict(X_test_norm.values)

    _id = [str(i) for i in range(1,(Y_test.shape[0]+1))]
    data = {'id':_id, 'label': Y_test.flatten().tolist()}
    output = pd.DataFrame(data)
    output.to_csv(path_or_buf=sys.argv[6], index=False)