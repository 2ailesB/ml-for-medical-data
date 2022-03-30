import os
os.system('which python')

from sklearn.model_selection import train_test_split
from preprocessing import formatting
from torch.utils.tensorboard import SummaryWriter
import datetime
from models import NN
from torch import nn
from models import classifier
from preprocessing import preprocessing
######################## STEP 1 : reading and preprocessing data ########################
# data reading
data = formatting.read_data()
print(f'data shape: {data.shape} | label: {data.iloc[:, -1].unique()}')

# Train test splitting
# As our dataset has balanced class number we simply shuffle it and split it
train_data, test_data = train_test_split(data, test_size=0.2, random_state=0, shuffle=True)
print(f'train data shape: {train_data.shape}')
print(f'test data shape: {test_data.shape}')
print(f'train data class number:\n {train_data[64].value_counts()}')
print(f'test data class number:\n {test_data[64].value_counts()}')

# LSTM = NN.LSTM_classifier2(train_data, test_data, input_size=8, num_layers=4, hidden_size=16, proj_size=4, dropout=0.2, final_activation=nn.Softmax())
# lr, epochs, batch_size = 1e-4, 100, 100
# start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# writer = SummaryWriter('figures/runs/LSTM_' + start_time )
# LSTM.fit(nn.CrossEntropyLoss(), lr, epochs, batch_size, writer)
# LSTM.save(lr, epochs, f'saved_models/LSTM', '')
# pred = LSTM.predict_proba(test_data)
# print(pred[0])

# LSTM = NN.LSTM_classifier2(train_data, test_data, input_size=8, num_layers=4, hidden_size=16, proj_size=4, dropout=0.2, final_activation=nn.Softmax(dim=1))
# LSTM.load('saved_models\LSTM\ckpt__epoch100.ckpt')
# # LSTM.predict_proba(test_data)
# LSTM.evaluation('LSTM')

model = classifier.MultiRFModel(train_data, test_data, preprocessing.basic_preprocessing())
model.evaluation('Random Forest_PR')