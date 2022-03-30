import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
import os
import time

from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from metrics.custom_metrics import accuracy


colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

# from preprocessing.sequences import table2seq


class NN_classifier(nn.Module):
    def __init__(self, train_data, test_data, n_in, layers, n_out, activation=nn.ReLU(), final_activation=None, p=0, batchnorm=True, preprocess=None) -> None:
        # https://stackoverflow.com/questions/46141690/how-to-write-a-pytorch-sequential-model
        super().__init__()

        self.preprocess = preprocess
        train_X, test_X = train_data.iloc[:, 0:64].to_numpy(), test_data.iloc[:, 0:64].to_numpy()
        if preprocess:
            train_X = self.preprocess.transform(train_X)
            test_X = self.preprocess.transform(test_X)

        self.train_X = torch.Tensor(train_X)
        self.train_y = torch.Tensor(train_data.iloc[:, 64].to_numpy())

        self.test_X = torch.Tensor(test_X)
        self.test_y = torch.Tensor(test_data.iloc[:, 64].to_numpy())

        layerlist = []
        for i in layers:
            # n_in input neurons connected to i number of output neurons
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(activation)  # Apply activation function - ReLU
            if batchnorm:
                # Apply batch normalization
                layerlist.append(nn.BatchNorm1d(i))
            if p:
                # Apply dropout to prevent overfitting
                layerlist.append(nn.Dropout(p))
            n_in = i  # Reassign number of input neurons as the number of neurons from previous last layer

        # Last layer
        layerlist.append(nn.Linear(layers[-1], n_out))
        if final_activation is not None:
            layerlist.append(final_activation)

        self.model = nn.Sequential(*layerlist)

        self.state = {}

    def forward(self, x):
        return self.model(x)

    def fit(self, loss, lr, epochs, batch_size, writer, test_loop=True):
        train_dataset = TensorDataset(
            self.train_X, self.train_y.to(torch.long))  # create your datset
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size)  # create your dataloader
        test_dataset = TensorDataset(
            self.test_X, self.test_y.to(torch.long))  # create your datset
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size)  # create your dataloader

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in tqdm(range(epochs)):
            losses = []
            for idx, (X, y) in enumerate(train_dataloader):
                yhat = self.model(X)
                optim.zero_grad()
                l = loss(yhat, y)
                l.backward()
                optim.step()
                losses.append(l)
            writer.add_scalar('train loss', torch.Tensor(losses).mean(), epoch)
            # writer.add_scalar('train accuracy', accuracy(yhat, y.reshape(y.shape[0], 1)), epoch)# TODO : add accuracy

            if test_loop:
                with torch.no_grad():
                    losses = []
                    for idx, (X, y) in enumerate(test_dataloader):
                        yhat = self.model(X)
                        l = loss(yhat, y)
                        losses.append(l)
                    writer.add_scalar('test loss', torch.Tensor(losses).mean(), epoch)
                    writer.add_scalar('test accuracy', self.score(accuracy), epoch) 

        return self.model

    def score(self, metrics):
        """Score on test dataset"""
        y_pred = self.predict()
        return metrics(y_pred, self.test_y)

    def predict(self):
        """Prediction with optimal parameters"""
        return torch.argmax(self.model(self.test_X), dim=1)

    def save(self, lr, n, ckpt_save_path, tag):
        self.state['lr'] = lr
        self.state['epoch'] = n
        self.state['state_dict'] = self.state_dict()
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        torch.save(self.state, os.path.join(
            ckpt_save_path, f'ckpt_{tag}_epoch{n}.ckpt'))

    def load(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.load_state_dict(state['state_dict'])

    def visualisation(self):
        pass

    def grid_search(self):
        pass

    def estimate_time(self):
        start_time = time.time()
        self.predict()
        ex_time = time.time() - start_time
        print(f'NN model test time: {ex_time}')


class LSTM_classifier2(nn.Module):
    """https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html"""
    def __init__(self, train_data, test_data, input_size, num_layers, hidden_size, proj_size=1, dropout=0, final_activation=None, preprocess=None) -> None:
        #  
        super().__init__()

        self.preprocess = preprocess
        train_X, test_X = train_data.iloc[:, 0:64].to_numpy(), test_data.iloc[:, 0:64].to_numpy()
        if preprocess:
            train_X = self.preprocess.transform(train_X)
            test_X = self.preprocess.transform(test_X)

        self.train_X = torch.Tensor(train_X).reshape(len(train_data), 8, 8) # N_sample, T,  N_feature(9342, 8, 8)
        self.train_y = torch.Tensor(train_data.iloc[:, 64].to_numpy())

        self.test_X =  torch.Tensor(test_X).reshape(len(test_data), 8, 8) #  N_sample, T,  N_feature(9342, 8, 8)
        self.test_y = torch.Tensor(test_data.iloc[:, 64].to_numpy())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, proj_size=proj_size, dropout=dropout, batch_first=True).to(self.device)
        self.final_activation = final_activation
        self.state = {}

    def forward(self, x):
        return self.lstm(x)[0][:,-1,:]  # get final output from last layer

    def fit(self, loss, lr, epochs, batch_size, writer, test_loop=True):
        train_dataset = TensorDataset(
            self.train_X, self.train_y.to(torch.long))  # create your datset
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, pin_memory=True)  # create your dataloader
        test_dataset = TensorDataset(
            self.test_X, self.test_y.to(torch.long))  # create your datset
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, pin_memory=True)  # create your dataloader

        optim = torch.optim.Adam(self.lstm.parameters(), lr=lr)
        for epoch in tqdm(range(epochs)):
            losses = []
            for idx, (X, y) in enumerate(train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                yhat = self.forward(X)
                optim.zero_grad()
                l = loss(yhat, y)
                l.backward()
                optim.step()
                losses.append(l)
            writer.add_scalar('train loss', torch.Tensor(losses).mean(), epoch)
            # TODO : add accuracy

            if test_loop:
                with torch.no_grad():
                    losses = []
                    for idx, (X, y) in enumerate(test_dataloader):
                        X = X.to(self.device)
                        y = y.to(self.device)
                        yhat = self.forward(X)
                        l = loss(yhat, y)
                        losses.append(l)
                    writer.add_scalar('test loss', torch.Tensor(losses).mean(), epoch)
                    writer.add_scalar('test accuracy', self.score(accuracy), epoch)

        return self.lstm

    def score(self, metrics):
        """Score on test dataset"""
        y_pred = self.predict().reshape((self.test_y.shape[0], 1))
        return metrics(y_pred, self.test_y.reshape((self.test_y.shape[0], 1)))

    def predict(self):
        """Prediction with optimal parameters"""
        X = self.test_X.to(self.device)
        return torch.argmax(self.lstm(X)[0][:, -1, :], dim=1)

    def predict_proba(self, x):
        """Prediction of probability for each class with softmax"""
        # x = torch.Tensor(x.iloc[:, 0:64].to_numpy()).reshape(len(x), 8, 8) # N_sample, T,  N_feature(9342, 8, 8)
        x = x.to(self.device)
        self.lstm.eval()
        return self.final_activation(self.forward(x)).detach().cpu()

    def save(self, lr, n, ckpt_save_path, tag):
        self.state['lr'] = lr
        self.state['epoch'] = n
        self.state['state_dict'] = self.lstm.state_dict()
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        torch.save(self.state, os.path.join(
            ckpt_save_path, f'ckpt_{tag}_epoch{n}.ckpt'))

    def load(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.lstm.load_state_dict(state['state_dict'])

    def visualisation(self):
        pass

    def grid_search(self):
        pass


    def evaluation(self, name):
        """Result evaluatio on test dataset with optimal parameters after grids search"""
        test_Y = label_binarize(self.test_y, classes=[0, 1, 2, 3])
        n_classes = test_Y.shape[1]
        y_score = self.predict_proba(self.test_X)

        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(test_Y[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(test_Y[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            test_Y.ravel(), y_score.ravel()
        )
        average_precision["micro"] = average_precision_score(test_Y, y_score, average="micro")
        self.show_test_pr_curves(recall, precision, average_precision, n_classes, name)


    def show_test_pr_curves(self, recall, precision, average_precision, n_classes, name):
        """ ref: 
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
        """
        _, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

        for i, color in zip(range(n_classes), colors):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title(f"Precision-Recall curves {name}")
        plt.savefig(f'./figures/{name}')
        plt.show()

    def estimate_time(self):
        start_time = time.time()
        self.predict()
        ex_time = time.time() - start_time
        print(f'LSTM model test time: {ex_time}')