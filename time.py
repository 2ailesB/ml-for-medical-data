import datetime
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from metrics.custom_metrics import accuracy, f1_micro

from preprocessing import inout
from preprocessing import formatting
from preprocessing import preprocessing

from models import classifier
from models import NN


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

    # preprocess = preprocessing.normPCA_preprocessing(n_components=10, kernel='rbf')
    preprocess = preprocessing.norm_preprocessing()
    preprocess.fit(train_data.iloc[:, 0:64])
    # preprocess=None
    ######################## STEP 2 : apply standard models ########################
    
    model2test = {'svm' : (classifier.SVMModel(train_data, test_data, preprocess), {"kernel": "rbf", "C": 100}),
                    'rf' : (classifier.RFModel(train_data, test_data, preprocess), {"n_estimators": 100, "criterion": "entropy", "max_depth":None}),
                    'logistic regression' : (classifier.LRModel(train_data, test_data, preprocess), {"penalty": 'l1', "C":0.1}), 
                    'lda': (classifier.lda(train_data, test_data, preprocess), {}), 
                    'qda': (classifier.qda(train_data, test_data, preprocess), {}), 
                    'sk_MLP': (classifier.sk_NN(train_data, test_data, preprocess), {'hidden_layer_sizes':(32, 16, 8), 'alpha':0.1, 'max_iter':100, 'batch_size':100}), 
                    'multi_rf':(classifier.MultiRFModel(train_data, test_data, preprocess), {})}

    for name, (model, param_grid) in model2test.items():
        t_time, e_time = model.estimate_time(param_grid)
        print(f'{name} model has training time {t_time} and execution time {e_time}')
    


if __name__ == '__main__':
    main()