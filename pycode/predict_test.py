import time
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.nn as nn

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

GPU = 1
TIMESTEP = 12
scaler_dict = dict()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        positional_embeddings = self.positional_embedding(torch.arange(0, TIMESTEP).to(self.device)).expand(1, TIMESTEP,
                                                                                                            self.dmodel).permute(
            1, 0, 2)
        input_embedding = z_embedding + positional_embeddings
        transformer_embedding = self.transformer_decoder(input_embedding, attention_mask)

        output = self.fc1(transformer_embedding.permute(1, 0, 2))
        output = self.fc12(output).permute(2, 0, 1)
        output = self.fc2(output)
        return output


def csv_to_tensor(file):
    Data = pd.read_csv(file)
    scaler = StandardScaler()
    scaler_dict[file.split('/')[2][0:-9]] = scaler
    f = Data['confirmedNum'].to_numpy()[::-1].reshape(42, 1).astype(float)
    f = scaler.fit_transform(f)
    x_data = np.arange(0, 42, 1)
    x_data = x_data.reshape(-1, 1).astype(int)
    data = np.append(x_data, values=f, axis=1)
    d = torch.from_numpy(data.copy())
    return d

def predict_day(model, name,days):
    model.eval()
    with torch.no_grad():
        x = days[:,0].reshape(1, -1).float()
        y = days[:,1].reshape(1, -1).float()
        attention_mask = torch.from_numpy(np.zeros((TIMESTEP, TIMESTEP)))
        y_pred = model(x, y, attention_mask)[0][0][0]
        return scaler_dict[name].inverse_transform(y_pred.cpu().reshape(-1,1))

model_dict = dict()
model_dict["beijing"] = Transformer();
model_dict["beijing"].load_state_dict(torch.load('./model/beijing.pt', map_location='cpu'))
model_dict["changchun"] = Transformer();
model_dict["changchun"].load_state_dict(torch.load('./model/changchun.pt', map_location='cpu'))
model_dict["shanghai"] = Transformer();
model_dict["shanghai"].load_state_dict(torch.load('./model/shanghai.pt', map_location='cpu'))
model_dict["shenzhen"] = Transformer();
model_dict["shenzhen"].load_state_dict(torch.load('./model/shenzhen.pt', map_location='cpu'))

# 预测自己城市后12天
def predict_city():
    for key in model_dict.keys():
        value = model_dict[key]
        print("正在预测城市：" + key)
        data = csv_to_tensor('./input/' + key + '_data.csv')
        pred = data.numpy().copy()
        result = np.zeros(shape=(12, 2))

        for i in range(12):
            a = predict_day(value, key, data[i: i + 12])
            pred[12+i][1] = a/1000;
            result[i][0] = i
            result[i][1] = a

#         这里理论上应该是测试准确率
        print("结果：")
        print(result)
        pd.DataFrame(result).to_csv('./predict/' + key + '_pred.csv')

# 预测其他城市后30天
def predict_other_city():
    for key in model_dict.keys():
        value = model_dict[key]
        print("正在利用模型：" + key + "预测其他城市")
        data = csv_to_tensor('./input/' + key + '_data.csv')
        pred = data.numpy().copy()
        result = np.zeros(shape=(12, 2))

        for city in model_dict.keys():
            if city == key:
                continue;
            data = csv_to_tensor('./input/' + city + '_data.csv')
            pred = data.numpy().copy()
            result = np.zeros(shape=(30, 2))

            for i in range(30):
                a = predict_day(value, key, data[i: i + 12])
                pred[12 + i][1] = a / 1000;
                result[i][0] = i
                result[i][1] = a

            print("结果：")
            print(result)
            pd.DataFrame(result).to_csv('./predict/' + key + '_to_pred_' + city + '.csv')

# predict_city()
predict_other_city()
