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
    model = classifier.SVMModel(train_data, test_data, preprocessing.basic_preprocessing, nfold=5)
    param_grid={"model__kernel": ["rbf"], "model__C": [1]}
    model.grid_search(param_grid)
    pred = model.predict()
    print(pred)
    print(model.score(accuracy))
    model.save('saved_models/SVM')


if __name__ == '__main__':
    main()