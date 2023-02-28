import datetime
import json
import sys
import os
import time
import traceback

import numpy as np
import pandas as pd
from joblib import dump
from sklearn import preprocessing
from sklearn.base import RegressorMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

data = {}
data['error'] = ''
data['accuracy'] = 0
data['metric'] = 'accuracy'
data['modelName'] = ""


def process(X_test, y_test, model):
    try:
        y_pred = model.predict(X_test)

        # Checking if model being evaluated is a classifier or regression
        is_regression = issubclass(type(model), RegressorMixin)

        # Evaluate regression models using R^2
        if is_regression:
            data['accuracy'] = r2_score(y_test, y_pred)
            data['metric'] = 'R-Squared'
        # Evaluate classification models using regular accuracy
        else:
            data['accuracy'] = float(np.average(y_test == y_pred))

        data['modelName'] = model.__class__.__name__

    except:
        data['error'] = str(traceback.format_exc())

    print(json.dumps(data))

    if not os.path.exists('out'):
        os.makedirs('out')
    dirname = os.path.dirname(__file__)

    file_name = "{:6.4f}".format(data['accuracy']).replace('.', '_') + '-' \
        + data['modelName'] + '-' \
        + str(int(time.time())) \
        + '.joblib'

    file_name_with_path = os.path.join(dirname, 'out', file_name)
    dump(model, file_name_with_path)


def k_neighbors_classifier(X, y, X_test, y_test):
    for i in range(3, 20):
        n_neighbors = i
        print('n_neighbors:', n_neighbors)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X, y)
        process(X_test, y_test, model)


def random_forest_classifier(X, y, X_test, y_test):
    for i in range(10, 20):
        for j in range(5, 20):
            depth = i
            estimators = j * 100
            print('depth:', depth, 'estimators:', estimators)
            model = RandomForestClassifier(random_state=0,
                                           n_estimators=estimators,
                                           max_depth=depth)
            model.fit(X, y)
            process(X_test, y_test, model)


def gradient_boost_classifier(X, y, X_test, y_test):
    for i in range(10, 20):
        for j in range(5, 20):
            depth = i
            estimators = j * 100
            print('depth:', depth, 'estimators:', estimators)
            model = GradientBoostingClassifier(random_state=0,
                                               n_estimators=estimators,
                                               max_depth=depth)
            model.fit(X, y)
            process(X_test, y_test, model)


def main():
    # Load train dataset
    train_df = pd.read_csv("./datasets/diabetes_training.csv")

    # Load test dataset
    X_test = np.load('./datasets/' + 'diabetes' + '.x.npy')
    y_test = np.load('./datasets/' + 'diabetes' + '.y.npy')
    test_df = pd.DataFrame(X_test, columns=list(train_df.columns)[0:-1])
    test_df.insert(len(test_df.columns), column='Outcome', value=y_test)

    print(train_df.head(10))
    print(test_df.head(10))

    # combine datasets
    train_df = pd.concat([train_df, test_df])

    # prepare for fitting
    X_train = train_df.drop(['Outcome'], axis=1)
    y_train = train_df['Outcome']

    print(X_train)
    print(y_train)

    # fit data
    k_neighbors_classifier(X_train, y_train, X_test, y_test)
    random_forest_classifier(X_train, y_train, X_test, y_test)
    gradient_boost_classifier(X_train, y_train, X_test, y_test)


main()
