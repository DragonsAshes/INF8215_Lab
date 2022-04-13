"""
Team:
<<<<< TEAM NAME >>>>>
Authors:
<<<<< NOM COMPLET #1 - MATRICULE #1 >>>>>
<<<<< NOM COMPLET #2 - MATRICULE #2 >>>>>
"""

BEANS = ['SIRA','HOROZ','DERMASON','BARBUNYA','CALI','BOMBAY','SEKER']

from bean_testers import BeanTester
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import *
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class MyBeanTester(BeanTester):
    def __init__(self):
        #self.randomforest = RandomForestClassifier(verbose=3, n_jobs=-1, max_depth=10, criterion='entropy')
        self.randomforest = RandomForestClassifier(verbose=3, n_jobs=-1, criterion='entropy', random_state=0)
        #self.randomforest = KNeighborsClassifier()
        #self.randomforest = linear_model.LogisticRegression(C=1e5)
        #self.randomforest = svm.SVC(kernel='linear')
        #self.randomforest = tree.DecisionTreeClassifier()
        #self.pca = PCA()
        self.pca = PCA(n_components=13)
        self.sc = StandardScaler()

    def train(self, X_train, y_train):
        # print("self : ", self)
        # print("X_train : ", X_train)
        # print("Y_train : ", y_train)
        """
        train the current model on train_data
        :param X_train: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :param y_train: 2D array of labels.
                each line is a different example.
                the first column is the example ID.
                the second column is the example label.
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        y_train = y_train[:,1:]

        X_train = X_train[:,1:]



        X_train = self.sc.fit_transform(X_train)
        X_train = self.pca.fit_transform(X_train)

        print(X_train[0])

        print(self.pca.explained_variance_ratio_)
            
        #X_train = np.delete(X_train[:,1:], [2,3], 1)
        self.randomforest.fit(X_train, y_train)


    def predict(self, X_data):
        """
        predict the labels of the test_data with the current model
        and return a list of predictions of this form:
        [
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            ...
        ]
        :param X_data: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 2D list of predictions with 2 columns: ID and prediction
        """
        X_data = np.array(X_data)
        X_data = X_data[:,1:]

        X_data = self.sc.transform(X_data)
        X_data = self.pca.transform(X_data)
        #X_data = np.delete(X_data, [2,3],1)
        predictions = self.randomforest.predict(X_data)
        print(predictions)
        
        return [[i+1, p] for i,p in enumerate(predictions)]
