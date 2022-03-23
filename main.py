from datetime import date
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import inout
from preprocessing import formatting
from preprocessing import preprocessing

from models import classifier

from metrics.scores import accuracy



def main():
    ######################## STEP 1 : reading and preprocessing data ########################
    # data reading
    data= formatting.read_data()
    print(f'data shape: {data.shape} | label: {data.iloc[:, -1].unique()}')

    # Train test splitting
    # As our dataset has balanced class number we simply shuffle it and split it
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=0, shuffle=True)
    print(f'train data shape: {train_data.shape}')
    print(f'test data shape: {test_data.shape}')
    print(f'train data class number:\n {train_data[64].value_counts()}')
    print(f'test data class number:\n {test_data[64].value_counts()}')

    ######################## STEP 2 : apply standard models ########################
    model2test = {'svm' : (classifier.SVMModel(train_data, test_data, preprocessing.basic_preprocessing), {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10]}),
                    'rf' : (classifier.random_forest_model(train_data, test_data, preprocessing.basic_preprocessing), {"model__n_estimators" : [100], "model__criterion": ['gini', 'entropy'], "model__max_depth":[2]}),
                    'logistic regression' : (classifier.logistic_regression(train_data, test_data, preprocessing.basic_preprocessing), {'model__penalty':['l1', 'l2'], 'model__C':[0.1, 1]}), 
                    'lda': (classifier.lda(train_data, test_data, preprocessing.basic_preprocessing), {}), 
                    'qda': (classifier.qda(train_data, test_data, preprocessing.basic_preprocessing), {})}

    for name, (model, param_grid) in model2test.items():
        model.grid_search(param_grid, n_fold=2)
        pred = model.predict()
        model.save(f'saved_models/{name}')
        model.visualisation(f'figures/{name}.png')
        print(f'{name} model after CV grid search has accuracy of {model.score(accuracy)}')

    # model.load('/home/yunfei/Desktop/ml-for-medical-data/saved_models/SVM')
    # model.load('saved_models/SVM')


if __name__ == '__main__':
    main()