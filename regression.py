import datetime
import json
import sys
import os
import time
import traceback

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import preprocessing
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
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


def random_forest_regressor(X, y, X_test, y_test):
    for i in range(3, 21):
        for j in range(15, 21):
            depth = i
            estimators = j * 100
            print('depth:', depth, 'estimators:', estimators)
            model = RandomForestRegressor(random_state=0,
                                          n_estimators=estimators,
                                          max_depth=depth,
                                          verbose=1,
                                          n_jobs=-1)
            model.fit(X, y)
            process(X_test, y_test, model)


def gradient_boost_regressor(X, y, X_test, y_test):
    for i in range(3, 20):
        for j in range(5, 20):
            depth = i
            estimators = j * 100
            print('depth:', depth, 'estimators:', estimators)
            model = GradientBoostingRegressor(random_state=0,
                                              n_estimators=estimators,
                                              max_depth=depth)
            model.fit(X, y)
            process(X_test, y_test, model)


def main():
    # Load train dataset
    train_df = pd.read_csv("./datasets/reg_housing_training.csv")

    # Load test dataset
    X_test = np.load('./datasets/' + 'HOUSING' + '.x.npy')
    y_test = np.load('./datasets/' + 'HOUSING' + '.y.npy')
    test_df = pd.DataFrame(X_test, columns=list(train_df.columns)[1:])
    test_df.insert(0, column='price', value=y_test)

    print(train_df.head(10))
    print(test_df.head(10))

    # combine datasets
    train_df = pd.concat([train_df, test_df])

    # prepare for fitting
    X_train = train_df.drop(['price'], axis=1)
    y_train = train_df['price']

    # fit data
    gradient_boost_regressor(X_train, y_train, X_test, y_test)
    random_forest_regressor(X_train, y_train, X_test, y_test)


main()
