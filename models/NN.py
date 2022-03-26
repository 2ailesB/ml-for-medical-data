import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.tensorboard
from torch.utils.data import TensorDataset, DataLoader

import os

from preprocessing.sequences import table2seq


class NN_classifier(nn.Module):
    def __init__(self, train_data, test_data, n_in, layers, n_out, activation=nn.ReLU(), final_activation=None, p=0, batchnorm=True) -> None:
        # https://stackoverflow.com/questions/46141690/how-to-write-a-pytorch-sequential-model
        super().__init__()

        self.train_X = torch.Tensor(train_data.iloc[:, 0:64].to_numpy())
        self.train_y = torch.Tensor(train_data.iloc[:, 64].to_numpy())

        self.test_X = torch.Tensor(test_data.iloc[:, 0:64].to_numpy())
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
        for epoch in range(epochs):
            losses = []
            for idx, (X, y) in enumerate(train_dataloader):
                yhat = self.model(X)
                optim.zero_grad()
                l = loss(yhat, y)
                l.backward()
                optim.step()
                losses.append(l)
            writer.add_scalar('train loss', torch.Tensor(losses).mean(), epoch)

            if test_loop:
                with torch.no_grad():
                    losses = []
                    for idx, (X, y) in enumerate(test_dataloader):
                        yhat = self.model(X)
                        l = loss(yhat, y)
                        losses.append(l)
                    writer.add_scalar('test loss', torch.Tensor(losses).mean(), epoch)

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


class LSTM_classifier(nn.Module):
    def __init__(self, train_data, test_data, n_in, n_layers, hiddens, n_out=None, activation=nn.ReLU(), final_activation=None, p=0) -> None:
        #  
        super().__init__()

        self.train_X = table2seq(torch.Tensor(train_data.iloc[:, 0:64].to_numpy()), 8, 8)
        self.train_y = torch.Tensor(train_data.iloc[:, 64].to_numpy())

        self.test_X = table2seq(torch.Tensor(test_data.iloc[:, 0:64].to_numpy()), 8, 8)
        self.test_y = torch.Tensor(test_data.iloc[:, 64].to_numpy())

        layerlist = []
        for n_l, hs in n_layers, hiddens:
            # n_in input neurons connected to i number of output neurons
            layerlist.append(nn.LSTM(input_size=n_in, num_layers=n_l, hidden_size=hs, dropout=p))


        # Last layer
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
        for epoch in range(epochs):
            losses = []
            for idx, (X, y) in enumerate(train_dataloader):
                yhat = self.model(X)
                optim.zero_grad()
                l = loss(yhat, y)
                l.backward()
                optim.step()
                losses.append(l)
            writer.add_scalar('train loss', torch.Tensor(losses).mean(), epoch)

            if test_loop:
                with torch.no_grad():
                    losses = []
                    for idx, (X, y) in enumerate(test_dataloader):
                        yhat = self.model(X)
                        l = loss(yhat, y)
                        losses.append(l)
                    writer.add_scalar('test loss', torch.Tensor(losses).mean(), epoch)

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


class LSTM_classifier2(nn.Module):
    def __init__(self, train_data, test_data, n_in, n_layers, hiddens, n_out=None, activation=nn.ReLU(), final_activation=None, p=0) -> None:
        #  
        super().__init__()

        self.train_X = torch.Tensor(train_data.iloc[:, 0:64].to_numpy()).reshape(len(train_data), 8, 8).permute((1, 0, 2)) # T, N_sample, N_feature(8, 9342, 8)
        self.train_y = torch.Tensor(train_data.iloc[:, 64].to_numpy())

        self.test_X = table2seq(torch.Tensor(test_data.iloc[:, 0:64].to_numpy()), 8, 8)
        self.test_y = torch.Tensor(test_data.iloc[:, 64].to_numpy())

        layerlist = []
        for n_l, hs in n_layers, hiddens:
            # n_in input neurons connected to i number of output neurons
            layerlist.append(nn.LSTM(input_size=n_in, num_layers=n_l, hidden_size=hs, dropout=p))


        # Last layer
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
        for epoch in range(epochs):
            losses = []
            for idx, (X, y) in enumerate(train_dataloader):
                yhat = self.model(X)
                optim.zero_grad()
                l = loss(yhat, y)
                l.backward()
                optim.step()
                losses.append(l)
            writer.add_scalar('train loss', torch.Tensor(losses).mean(), epoch)

            if test_loop:
                with torch.no_grad():
                    losses = []
                    for idx, (X, y) in enumerate(test_dataloader):
                        yhat = self.model(X)
                        l = loss(yhat, y)
                        losses.append(l)
                    writer.add_scalar('test loss', torch.Tensor(losses).mean(), epoch)

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