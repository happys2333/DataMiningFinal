import time
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.nn as nn

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.autograd import Variable

GPU = 0
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")

# Param list
EPOCH = 200
TIMESTEP = 12
TRAINDAYS = 30

OPTIMIZER = 'Adam'
LEARN = 0.0005
PATIENCE = 15
beijing = None
changchun = None
shenzhen = None
shanghai = None



def csv_to_tensor(file):
    Data = pd.read_csv(file)
    f = Data['confirmedNum'].to_numpy()[::-1].reshape(42, 1).astype(float)
    x_data = np.arange(0, 42, 1)
    x_data = x_data.reshape(-1, 1).astype(int)
    data = np.append(x_data, values=f, axis=1)
    d = torch.from_numpy(data.copy())
    return d


# Util function
def read_csv_to_list():
    global beijing
    global shenzhen
    global changchun
    global shanghai
    beijing = csv_to_tensor('../input/covid-19-datamining/beijing_data.csv')
    shenzhen = csv_to_tensor('../input/covid-19-datamining/shenzhen_data.csv')
    changchun = csv_to_tensor('../input/covid-19-datamining/changchun_data.csv')
    shanghai = csv_to_tensor('../input/covid-19-datamining/shanghai_data.csv')


read_csv_to_list()


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self, in_channels=1, embedding_size=256, k=5):
        super(context_embedding, self).__init__()
        self.causal_convolution = CausalConv1d(in_channels, embedding_size, kernel_size=k)

    def forward(self, x):
        x = self.causal_convolution(x)
        return torch.tanh(x)


class Transformer(nn.Module):
    def __init__(self, device=device, dmodel=256):
        super(Transformer, self).__init__()
        self.input_embedding = context_embedding(2, dmodel, 9)
        self.positional_embedding = torch.nn.Embedding(TIMESTEP, dmodel)
        self.device = device
        self.dmodel = dmodel
        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=dmodel, nhead=8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=6)
        self.fc1 = torch.nn.Linear(dmodel, int(dmodel / 2))
        self.fc12 = torch.nn.Linear(int(dmodel / 2), 1)
        self.fc2 = torch.nn.Linear(TIMESTEP, 1)

    def forward(self, x, y, attention_mask):

        z = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)
        positional_embeddings = self.positional_embedding(torch.arange(0, TIMESTEP).to(self.device)).expand(1,TIMESTEP,self.dmodel).permute(1,0,2)
        input_embedding = z_embedding + positional_embeddings
        transformer_embedding = self.transformer_decoder(input_embedding, attention_mask)

        output = self.fc1(transformer_embedding.permute(1, 0, 2))
        output = self.fc12(output).permute(2,0,1)
        output = self.fc2(output)
        return output


# def ToVariable(x):
#     tmp = torch.FloatTensor(x)
#     return Variable(tmp)
# class LSTMpred(nn.Module):
#     def __init__(self,input_size,hidden_dim):
#         super(LSTMpred,self).__init__()
#         self.input_dim = input_size
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(input_size,hidden_dim)
#         self.hidden2out = nn.Linear(hidden_dim,1)
#         self.hidden = self.init_hidden()
#         self.fc = nn.Linear(12,1)
#     def init_hidden(self):
#         return (Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
#                 Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))
#     def forward(self,seq):
#         lstm_out, self.hidden = self.lstm(
#             seq.view(len(seq), 1, -1), self.hidden)
#         outdat = self.hidden2out(lstm_out.view(len(seq),-1)).permute(1,0)
#         outdat = self.fc(outdat)
#         return outdat


Loss = nn.L1Loss()
model = Transformer().cuda(device=device)


# model = LSTMpred(1,6).cuda(device=device)


def evaluate_epoch(model, test):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, TIMESTEP):
            x = test[i + TRAINDAYS - TIMESTEP: i + TRAINDAYS, 0].reshape(1, -1).float()
            y = test[i + TRAINDAYS - TIMESTEP: i + TRAINDAYS, 1].reshape(1, -1).float()
            attention_mask = torch.from_numpy(np.zeros((TIMESTEP, TIMESTEP)))
            y_pred = model(x.cuda(),y.cuda(), attention_mask.cuda())[0][0][0]
            # y_pred = model(x.cuda())[0][0]
            loss = Loss(y_pred.float(), test[i + TRAINDAYS][1].cuda().float())
            l_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def train(model,name, data):
    print('Model Training Started ...', time.ctime())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    min_val_loss = np.inf
    for epoch in range(EPOCH):
        starttime = datetime.now()
        model.train()
        loss_sum, n = 0.0, 0
        for i in range(0, TRAINDAYS - TIMESTEP):
            optimizer.zero_grad()
            x = data[i:i + TIMESTEP, 0].reshape(1, -1).float()
            y = data[i:i + TIMESTEP, 1].reshape(1, -1).float()
            attention_mask = torch.from_numpy(np.zeros((TIMESTEP, TIMESTEP)))
            y_pred = model(x.cuda(),y.cuda(), attention_mask.cuda())[0][0][0]
            # model.hidden = model.init_hidden()
            # y_pred = model(x.cuda())[0][0]
            loss = Loss(y_pred.float(), data[i + TIMESTEP][1].cuda().float())
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n += 1
        train_loss = loss_sum / n
        val_loss = evaluate_epoch(model, data)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), './model/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time, " seconds ", "train loss:", train_loss, "validation loss:",
              val_loss)
    print('Model Training Ended ...', time.ctime())


def pred_value(model, x, y):
    model.eval()
    with torch.no_grad():
        attention_mask = torch.from_numpy(np.zeros((TIMESTEP, TIMESTEP)))
        y_pred = model(x.cuda(),y.cuda(), attention_mask.cuda())[0][0][0]
    return y_pred



train(model,'shenzhen', shenzhen)


def test(model,name, data):
    loss = nn.L1Loss()
    print('Model Testing Started ...', time.ctime())
    model.load_state_dict(torch.load('./model/' + name + '.pt'))

    print('Model Testing Ended ...', time.ctime())