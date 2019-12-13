import numpy as np
import pandas as pd
import spacy
import pickle
from gensim.models import Word2Vec
import re
from MyStemmer import MyPorterStemmer
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import random
# import matplotlib.pyplot as plt
import sys

def preprocess(string):
    string = string.lower()
    regex1 = re.compile('@user')
    regex2 = re.compile('@user')
    string = regex1.sub("", string)
    string = regex2.sub("", string)
    for same_char in re.findall(r'((\w)\2{2,})', string):
        string = string.replace(same_char[0], same_char[1])
    for digit in re.findall(r'\d+', string):
        string = string.replace(digit, "1")
    for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
        if punct[0:2] =="..":
            string = string.replace(punct, "...")
        else:
            string = string.replace(punct, punct[0])
    regex3 = re.compile('[^a-zA-Z0-9.,;:?!\' ]')
    string = regex3.sub("", string)
    return string

def tokenization(train_x):
    ret = []
    sp = spacy.load('en_core_web_sm')
    p = MyPorterStemmer()
    for sen in train_x:
        sen = preprocess(sen)
        word = sp(sen)
#         tokens = [token.text for token in word]
        tokens = [p.stem(token.text) for token in word]
        ret.append(tokens)
    
    return ret

def get_set(x, y=None, test=False):
    
    if not test:
        dataset = list(zip(x,y))
        random.shuffle(dataset)

        train_set = dataset[:11264]
        valid_set = dataset[11264:]

        return train_set, valid_set
    else:
        _y = [0 for i in range(len(x))]
        return list(zip(x,_y))
        

class myDataset(Dataset):
    
    def __init__(self, data, w2v_model=None):
        self.w2v_model = w2v_model
#         self.test = test
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_x = self.data[index][0]
        data_x = list(map(lambda x: self.w2v_model.wv[x], data_x))
        data_x = torch.tensor(data_x)
        
#         if self.test:
#             return data_x
        
        data_y = torch.tensor(self.data[index][1])
        return (data_x, data_y)

def myCollate_fn(sample):
    
    sample.sort(key=lambda x: len(x[0]), reverse=True)
    tokens = [x[0] for x in sample]
    labels = torch.stack([x[1] for x in sample])
    lens = [len(x[0]) for x in sample]
    
    pad_tokens = pad_sequence(tokens, batch_first=True)
    pack_pad_tokens = pack_padded_sequence(pad_tokens, lens, batch_first=True)
    
    return pack_pad_tokens, labels

class myNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(myNet, self).__init__()
        
#         self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True, dropout=0.2, bidirectional=True)
#         self.gru2 = nn.GRU(hidden_size*2, hidden_size, batch_first=True, dropout=0.35, bidirectional=True)
        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.relu = nn.ReLU()
                
        self.linear = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.Sigmoid(),
#             nn.ReLU(),
#             nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Sigmoid(),
#             nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
    
    def forward(self, x, h=None):
#         x, h = self.gru1(x, h)
#         x, h = self.gru2(x, h)
        x, h = self.gru(x, h)
        h = torch.cat((h[0], h[1]),1)
        output = self.relu(h)
        output = self.linear(output)
        
        return output


class myRNN():
    
    def __init__(self, epochs=10, lr=1e-4, batch_size=128, input_size=100, hidden_size=128, n_layers=2, dropout=0.2):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.net = None
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.loss_list = []
        self.accu_list = []
        self.recall_list = []
        self.val_accu_list = []
        self.val_recall_list = []
        self.val_f1_list = []
        self.f1_list = []
        self.gpu = torch.cuda.is_available()
        
        
    def predict(self, vec):
        out = F.softmax(vec, dim=1)
        _, idx = out.max(-1)
        return idx.view(-1, 1)
    
    def accu_cnt(self, y_pred, y_true):
        return sum([1 if y_pred[i] == y_true[i] else 0 for i in range(len(y_pred))])
    
    def recall_cnt(self, y_pred, y_true):
        return sum([1 if (y_pred[i] == y_true[i] and y_pred[i] == 1) else 0 for i in range(len(y_pred))])
    
    def get_loss(self):
        return self.loss_list
    
    def get_accu(self):
        return self.accu_list
    
    def get_recall(self):
        return self.recall_list
    
    def get_f1(self):
        return self.f1_list
    
    def get_valid_accu(self):
        return self.val_accu_list
    
    def get_valid_recall(self):
        return self.val_recall_list
    
    def get_valid_f1(self):
        return self.val_f1_list
    
    def fit(self, trainset, validset, verbose=True):
        if self.gpu:
            device = 0
        max_val_accu = 0
        net = myNet(self.input_size, self.hidden_size, self.n_layers, self.dropout)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        if self.gpu:
            net.cuda(device)
        for epoch in range(self.epochs):
            tmp_cnt = 0
            tmp_recall = 0
            N = 0
            N_recall = 0
            total_loss = 0
            net.train()
            
            for i, (x, y) in enumerate(trainset):
                if self.gpu:
                    x = x.cuda(device)
                    y = y.cuda(device)
                preds = net(x, None)
                if self.gpu:
                    preds = preds.cuda(device)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                preds_labels = self.predict(preds).flatten()
                tmp_cnt += self.accu_cnt(preds_labels, y)
                tmp_recall += self.recall_cnt(preds_labels, y)
                
                N += len(y)
                N_recall += sum(y).item()
                
                total_loss += loss.item()*len(y)
                if verbose:
                    print(f'Epoch: {epoch+1}/{self.epochs}, Batch {i+1}, Loss: {loss.item()}', end='\r')
            
            train_loss = total_loss/N
            train_accu = tmp_cnt/N
            train_recall = tmp_recall/N_recall
            f1_score = (2*train_accu*train_recall)/(train_accu+train_recall)
            self.loss_list.append(train_loss)
            self.accu_list.append(train_accu)
            self.recall_list.append(train_recall)
            self.f1_list.append(f1_score)
            if verbose:
                print(f'---Epoch: {epoch+1}/{self.epochs}, Loss: {round(train_loss,4)}, Train accuracy: {round(train_accu,4)}, Train_recall: {round(train_recall,4)}, F1 score: {round(f1_score,4)}---')
            
            # Validation
            
            valid_cnt = 0
            valid_recall = 0
            N_valid = len(validset)
            N_val_recall = 0
            net.eval()
            for i, (x, y) in enumerate(validset):
                if self.gpu:
                    x = x.cuda(device)
                    y = y.cuda(device)
                preds = net(x)
                if self.gpu:
                    preds = preds.cuda(device)
                preds_labels = self.predict(preds).flatten()
                if preds_labels.item() == y:
                    valid_cnt += 1
                    if y == 1:
                        valid_recall += 1
                if y == 1:
                    N_val_recall += 1
            valid_accu = valid_cnt/N_valid
            valid_recall = valid_recall/N_val_recall
            val_f1 = (2*valid_recall*valid_accu)/(valid_recall+valid_accu)
            self.val_accu_list.append(valid_accu)
            self.val_recall_list.append(valid_recall)
            self.val_f1_list.append(val_f1)
            if verbose:
                print(f'---Validation accuracy: {round(valid_accu,4)}, Validation recall: {round(valid_recall,4)}, Validation F1 score: {round(val_f1,4)}---')
            if valid_accu > max_val_accu:
                max_val_accu = valid_accu
                tmp_model = net
            
        
        self.net = tmp_model
        if verbose:
            print(f'Max validation accuracy: {max_val_accu}')
        
    def transform(self, data):
        ret = []
        self.net.eval()
        for i, (x, _) in enumerate(data):
            if self.gpu:
                device = 0
                x = x.cuda(device)
            preds = self.net(x)
            if self.gpu:
                preds = preds.cuda(device)
            preds_labels = self.predict(preds).flatten()
            
            ret.append(preds_labels.item())
        return ret


if __name__=="__main__":
    use_gpu = torch.cuda.is_available()

    train_x = pd.read_csv(sys.argv[1])
    train_y = pd.read_csv(sys.argv[2])
    test_x = pd.read_csv(sys.argv[3])

    train_x_v = train_x["comment"].values
    test_x_v = test_x["comment"].values
    train_label = train_y['label'].values.tolist()

    train_x_token = tokenization(train_x_v)
    test_x_token = tokenization(test_x_v)

    train_set, valid_set = get_set(train_x_token, train_label)

    epochs = 50
    lr = 1e-4
    batch_size = 128
    input_size = 250
    hidden_size = 128
    n_layers = 3
    dropout = 0.8

    all_data = train_x_token + test_x_token
    w2v_model = Word2Vec(all_data, size=input_size, window=5, min_count=1, workers=4)
    w2v_model.train(all_data, total_examples= w2v_model.corpus_count, epochs= 300)
    w2v_model.save(f'./w2v.model')

    trainset = myDataset(train_set, w2v_model)
    validset = myDataset(valid_set, w2v_model)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=myCollate_fn)
    valid_loader = DataLoader(validset, batch_size=1, collate_fn=myCollate_fn)

    myrnn = myRNN(epochs, lr, batch_size, input_size, hidden_size, n_layers, dropout)
    myrnn.fit(train_loader, valid_loader)

    with open(f'./model_best.pkl', 'wb') as output:
        pickle.dump(myrnn, output)
    
    # with open(f'./model_best.pkl', 'rb') as file:
    #     myrnn = pickle.load(file)

    # test_set = get_set(test_x_token, test=True)
    # testset = myDataset(test_set, w2v_model)
    # test_loader = DataLoader(testset, batch_size=1, collate_fn=myCollate_fn)

    # ans = myrnn.transform(test_loader)

    # result = pd.DataFrame()
    # result['id'] = [i for i in range(860)]
    # result['label'] = ans
    # result.to_csv(sys.argv[4], index=False)