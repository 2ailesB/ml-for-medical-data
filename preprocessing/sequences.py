import torch

def table2seq(seq, lseq, dim):
    return seq.reshape(lseq, seq.shape[0], dim)