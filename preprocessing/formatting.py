import numpy as np
import pandas as pd
import os

from .inout import *


def merge_np(data1, data2, data3, data4):
    res = np.concatenate((data1, data2, data3, data4), axis=0)
    return res

def merge_pd(data1, data2, data3, data4):
    res = pd.concat((data1, data2, data3, data4), axis=0)
    return res

def data_fusion(data_path, name):
    data1, data2, data3, data4 = (csv2pd(f'{data_path}/{i}.csv') for i in range(4)) # (2910, 65) 
    data = merge_pd(data1, data2, data3, data4) # (11678, 65)
    save_path = os.path.join(data_path, name)
    if not os.path.exists(save_path):
        pd2csv(data, save_path)
    return data

def read_data(data_path='./data', name='data_full.csv'):
    path = os.path.join(data_path, name)
    if not os.path.exists(path):
        data = data_fusion(data_path, name)
    else:
        data = pd.read_csv(path, sep=',',header=None)
    return data
