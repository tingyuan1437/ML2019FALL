import os
import pandas as pd
import numpy as np
import re
import math
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import sys

def readdata(data):
    
    for col in list(data.columns[2:]):
        data[col] = data[col].astype(str).map(lambda x: x.rstrip('x*#A'))
    data = data.values

    data = np.delete(data, [0,1], 1)

    data[ data == 'NR'] = 0
    data[ data == ''] = 0
    data[ data == 'nan'] = 0
    data = data.astype(np.float)

    return data

def extract(data):
    N = data.shape[0] // 18

    temp = data[:18, :]
    
    for i in range(1, N):
        temp = np.hstack((temp, data[i*18: i*18+18, :]))
    return temp

def valid(x, y):
    if y <= 2 or y > 75:
        return False
    for i in range(9):
        if x[0,i] < 5:
            return False
        if x[1,i] < 1 or x[1,i] > 2.5:
            return False
        if x[2,i] < 0 or x[2,i] > 2:
            return False
        if x[3,i]  > 1:
            return False
        if x[4,i] < -1 or x[4,i] > 75:
            return False
        if x[5,i] < -1 or x[5,i] > 70:
            return False
        if x[6,i] > 110:
            return False
        if x[7,i] > 125:
            return False
        if x[8,i] < -5 or x[8,i] > 300:
            return False
        if x[9,i] <= 2 or x[9,i] > 75:
            return False
        if x[10,i] > 40:
            return False
        if x[11,i] < 20:
            return False
        if x[12,i] > 25:
            return False
        if x[13,i] < 1 or x[13,i] > 3.5:
            return False
        if x[16,i] > 7:
            return False
        if x[17,i] > 7.5:
            return False
    return True

def parse2train(data):
    x = []
    y = []

    total_length = data.shape[1] - 9
    for i in range(total_length):
        x_tmp = data[:,i:i+9]
        y_tmp = data[9,i+9]
        if valid(x_tmp, y_tmp):
            x.append(x_tmp.reshape(-1,))
            y.append(y_tmp)
    x = np.array(x)
    y = np.array(y)
    return x,y

def parseTest(data):
    x = []
    for i in range(500):
        x_tmp = data[i*18:i*18+18, :].reshape(-1,)
        x.append(x_tmp)
    return np.array(x)

class linearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out



if __name__ == "__main__":
    # x_y1 = pd.read_csv("year1-data.csv")
    # x_y2 = pd.read_csv("year2-data.csv")

    # x = x_y1.append(x_y2)
    
    # twoyear = readdata(x)
    # train_data = extract(twoyear)
    # train_x, train_y = parse2train(train_data)
    
    # learningRate = 1e-4
    # epochs = 1000
    # BATCH_SIZE = 64
    # LOSS = []

    # model = linearRegression(train_x.shape[1], 1)

    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    
    # inputs = Variable(torch.from_numpy(train_x).float())
    # labels = Variable(torch.from_numpy(train_y).float())
    
    # torch_dataset = TensorDataset(inputs, labels)
    
    # loader = DataLoader(
    #     dataset=torch_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=2,
    # )
    
    # for epoch in range(epochs):
    #     for step, (batch_x, batch_y) in enumerate(loader):
    #         optimizer.zero_grad()
    #         outputs = model(batch_x)
    #         loss = criterion(outputs.reshape(-1,1), batch_y.reshape(-1,1))
            
    #         LOSS.append(loss.item())
            
    #         loss.backward()
    #         optimizer.step()
            
    #     print('Epoch {}, Loss {}'.format(epoch, loss.item()), end = '\r')

    # torch.save(model.state_dict(), './model_best_v3_5.5441')

    test_data = pd.read_csv(sys.argv[1])
    test = readdata(test_data)
    test_x = parseTest(test)

    model = linearRegression(test_x.shape[1], 1)
    model.load_state_dict(torch.load('./model_best_v3_5.5441'))
    test_y = model(Variable(torch.from_numpy(test_x).float()))

    _id = ['id_'+str(i) for i in range(500)]
    data = {'id':_id, 'value': test_y.flatten().tolist()}
    output = pd.DataFrame(data)

    output.to_csv(path_or_buf=sys.argv[2], index=False)