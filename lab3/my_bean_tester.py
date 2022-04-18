"""
Team:
Kazou
Authors:
Virgile Retault - 2164296
Sebastien Foucher - 2162248
"""

BEANS = ['SIRA','HOROZ','DERMASON','BARBUNYA','CALI','BOMBAY','SEKER']

from bean_testers import BeanTester
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class MyBeanTester(BeanTester):
    def __init__(self):
        # Random Forest Classifier
        self.randomforest = RandomForestClassifier(verbose=3, n_jobs=-1, criterion='entropy')
        # Principal Components Analysis with 12 components
        self.pca = PCA(n_components=12)
        # Standard Scaler
        self.sc = StandardScaler()

    def train(self, X_train, y_train):
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
        # Prepare the data by removing the ID column
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        y_train = y_train[:,1]
        X_train = X_train[:,1:]

        # Train and apply standard scaler and PCA
        X_train = self.sc.fit_transform(X_train)
        X_train = self.pca.fit_transform(X_train)

        # Train the model
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
        # Prepare the data by removing the ID column
        X_data = np.array(X_data)
        X_data = X_data[:,1:]

        # Apply standard scaler and PCA
        X_data = self.sc.transform(X_data)
        X_data = self.pca.transform(X_data)

        # Make the predictions 
        predictions = self.randomforest.predict(X_data)

        # Add back the ID column and return
        return [[i+1, p] for i,p in enumerate(predictions)]
