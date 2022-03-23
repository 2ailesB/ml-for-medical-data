import torch
import torch.nn.functional as F
import torch.nn as nn

import os


class NN_classifier():
    def __init__(self, n_in, layers, n_out, activation = nn.ReLU(), final_activation = None, p=0, batchnorm=True) -> None:
        # https://stackoverflow.com/questions/46141690/how-to-write-a-pytorch-sequential-model
        layerlist = []
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))  # n_in input neurons connected to i number of output neurons
            layerlist.append(activation)  # Apply activation function - ReLU
            if batchnorm:
                layerlist.append(nn.BatchNorm1d(i))  # Apply batch normalization
            if p:
                layerlist.append(nn.Dropout(p))  # Apply dropout to prevent overfitting
            n_in = i  # Reassign number of input neurons as the number of neurons from previous last layer

        # Last layer
        layerlist.append(nn.Linear(layers[-1], n_out))
        if final_activation is not None:
            layerlist.append(final_activation)

        self.model = nn.Sequential(*layerlist)

    def forward(self, x):
        return self.model(x)

    def fit(self, train_data, loss, lr, epochs):
        optim = torch.optim.Adam(self.model.parameters(), lr = lr)
        for epoch in range(epochs):
            for X, y in train_data:
                yhat = self.model(X)
                optim.zero_grad()
                l = loss(yhat, y)
                optim.backward()
                optim.step()
            
        return self.model

    def save(self, lr, n, ckpt_save_path, tag):
        self.state['lr'] = lr
        self.state['epoch'] = n
        self.state['state_dict'] = self.state_dict()
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        torch.save(self.state, os.path.join(ckpt_save_path, f'ckpt_{tag}_epoch{n}.ckpt'))

    def load(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.load_state_dict(state['state_dict'])



