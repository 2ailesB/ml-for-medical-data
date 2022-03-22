from email import header
import pandas as pd
import numpy as np

def csv2pd(path):
    return pd.read_table(path, sep=',', decimal='.', header=None)

def csv2np(path):
    return pd.read_table(path, sep=',', decimal='.', header=None).to_numpy()

def np2pd(data):
    return pd.DataFrame(data)

def pd2np(data):
    return data.to_numpy()

def np2csv(data, path):
    np.savetxt(path, data, delimiter=",")
    return path

def pd2csv(data, path):
    data.to_csv(path, columns=None, header=False, index=False)
    return path

if __name__ == '__main__':
    d = csv2pd('data/0.csv')
    print(d.shape)
    pd2csv(d, 'tests/testpd.csv')
    dp = pd2np(d)
    print(dp.shape)
    np2csv(dp, 'tests/testnp.csv')
    dp2 = csv2np('tests/testnp.csv')
    print(dp2.shape)
    d2 = csv2pd('tests/testpd.csv')
    print(d2.shape)