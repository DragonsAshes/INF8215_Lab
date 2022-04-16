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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical


parameters = {
    "n_estimators":[5,10,50,100,250],
    "max_depth":[2,4,8,16,32,None],
    "learning_rate":[1.0]
    
}
def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')

class MyBeanTester(BeanTester):
    def __init__(self):
        #self.randomforest = RandomForestClassifier(verbose=3, n_jobs=-1, max_depth=10, criterion='entropy')
        self.randomforest = RandomForestClassifier(verbose=3, n_jobs=-1, criterion='entropy')
        #self.randomforest = ExtraTreesClassifier(min_samples_split=2)
        #self.randomforest = GradientBoostingClassifier(n_estimators=250, learning_rate=1.0, max_depth=10, random_state=0)
        self.ada = AdaBoostClassifier(base_estimator=self.randomforest, learning_rate=1.0, n_estimators=250)
        #self.randomforest = GridSearchCV(GradientBoostingClassifier,parameters,cv=5, n_jobs=-1)
        #self.randomforest = KNeighborsClassifier()
        #self.randomforest = linear_model.LogisticRegression(C=1e5)
        #self.randomforest = svm.SVC(kernel='linear')
        #self.randomforest = tree.DecisionTreeClassifier()
        #self.pca = PCA()
        self.pca = PCA(n_components=12)
        self.sc = StandardScaler()

                # Initialising the ANN
        self.classifier = Sequential()
        self.classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
        self.classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
        self.classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax'))
        self.classes = ['SEKER', 'HOROZ', 'SIRA', 'DERMASON', 'BARBUNYA', 'CALI', 'BOMBAY']


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
        
        y_train = y_train[:,1]

        X_train = X_train[:,1:]


        y_train = np.array([self.classes.index(a) for a in y_train])
        print(y_train)
        y_train = to_categorical(y_train)
        print(y_train)
        print(X_train)

        X_train = self.sc.fit_transform(X_train)
        X_train = self.pca.fit_transform(X_train)

        #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=0)
            
        #X_train = np.delete(X_train[:,1:], [2,3], 1)
        self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        self.classifier.fit(X_train, y_train, batch_size = 10, epochs = 200)
        # a = self.randomforest.fit(X_train, y_train)
        # print("score : ",a.score(X_test, y_test))
        #display(self.randomforest)


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
        predictions = self.classifier.predict(X_data)

        predictions = [self.classes[np.argmax(a)] for a in predictions]
        # predictions = self.randomforest.predict(X_data)
        print(predictions)
        
        return [[i+1, p] for i,p in enumerate(predictions)]
