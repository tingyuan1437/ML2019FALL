import os
import sys
import random
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import PIL
import matplotlib.pyplot as plt

def load_data(img_path, label_path):
    train_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    train_label = pd.read_csv(label_path)
    train_label = train_label.iloc[:,1].values.tolist()
    
    t1 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    t2 = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=1),
        torchvision.transforms.ToTensor(),
    ])
    t3 = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(degrees=(15,15)),
        torchvision.transforms.ToTensor(),
    ])
    t4 = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(degrees=(-15,-15)),
        torchvision.transforms.ToTensor(),
    ])

    dataset = []

    for i, image in enumerate(train_image):
        img = Image.open(image)
        img_1 = t1(img)
        img_2 = t2(img)
        img_3 = t3(img)
        img_4 = t4(img)
        dataset.append((img_1, train_label[i]))
        dataset.append((img_2, train_label[i]))
        dataset.append((img_3, train_label[i]))
        dataset.append((img_4, train_label[i]))    
    
    random.shuffle(dataset)
    
    train_set = dataset[:100000]
    valid_set = dataset[100000:]
    
    return train_set, valid_set

class hw3_dataset(Dataset):
    
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx][0]
        label = self.data[idx][1]
        return img, label

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),     
            nn.Dropout(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),        
            nn.Dropout(0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.35),
            
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.35),
        )
        self.fc = nn.Sequential(
            nn.Linear(3*3*256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        #image size (48,48)
        x = self.conv1(x) #(24,24)
        x = self.conv2(x) #(12,12)
        x = self.conv3(x) #(6,6)
        x = self.conv4(x) #(3,3)
        x = x.view(-1, 3*3*256)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()

    train_set, valid_set = load_data(sys.argv[1], sys.argv[2])

    train_dataset = hw3_dataset(train_set)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    valid_dataset = hw3_dataset(valid_set)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    model = Net()
    # model.load_state_dict(torch.load('./model_d_v2'))
    
    if use_gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    train_loss_history = []
    train_accu_history = []
    valid_loss_history = []
    valid_accu_history = []

    num_epoch = 1100
    for epoch in range(num_epoch):
        model.train()
        train_loss = []
        train_acc = []
        for idx, (img, label) in enumerate(train_loader):
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        ta = np.mean(train_acc)
        tl = np.mean(train_loss)
        train_accu_history.append(ta)
        train_loss_history.append(tl)
        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, tl, ta))


        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            for idx, (img, label) in enumerate(valid_loader):
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                output = model(img)
                loss = loss_fn(output, label)
                predict = torch.max(output, 1)[1]
                acc = np.mean((label == predict).cpu().numpy())
                valid_loss.append(loss.item())
                valid_acc.append(acc)
            va = np.mean(valid_acc)
            vl = np.mean(valid_loss)
            valid_accu_history.append(va)
            valid_loss_history.append(vl)
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, vl, va))
        
#         if np.mean(train_acc) > 0.99:
#             checkpoint_path = './model_d/model_handcraft_do_{}.pth'.format(epoch+1) 
#             torch.save(model.state_dict(), checkpoint_path)
#             print('model saved to %s' % checkpoint_path)

    torch.save(model.state_dict(), './model_d_v2')
