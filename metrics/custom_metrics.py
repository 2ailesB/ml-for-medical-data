from sklearn.metrics import f1_score

def accuracy(y_pred, y_true):
    return (y_pred==y_true).sum()/y_pred.shape[0]

def f1_micro(y_pred, y_true):
    return f1_score(y_true, y_pred, average='micro')