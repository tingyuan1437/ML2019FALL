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

    loss_fn = nn.CrossEntropyLoss()
    
    model = Net()
    model.load_state_dict(torch.load('./model_d_v2'))
    model.eval()
    
    if use_gpu:
        model.cuda()

    test_path = sys.argv[1]

    test_image = sorted(glob.glob(os.path.join(test_path, '*.jpg')))
    
    t1 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
    test_img = []

    for i, image in enumerate(test_image):
        img = Image.open(image)
        img_1 = t1(img)
        test_img.append(img_1)
    
    _ = [0 for i in range(len(test_img))]
    test_set = list(zip(test_img, _))
    
    test_dataset = hw3_dataset(test_set)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    y_pred = []
    
    with torch.no_grad():
        for idx, (img, label) in enumerate(test_loader):
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                output = model(img)
                loss = loss_fn(output, label)
                predict = torch.max(output, 1)[1]
                y_pred += predict.tolist()
        
    _id = [str(i) for i in range(len(y_pred))]
    data = {'id':_id, 'label': y_pred}
    output = pd.DataFrame(data)

    output.to_csv(path_or_buf=sys.argv[2], index=False)