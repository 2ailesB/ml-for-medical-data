def accuracy(y_pred, y_true):
    return (y_pred==y_true).sum()/y_pred.shape[0]