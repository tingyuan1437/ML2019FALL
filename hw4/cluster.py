import numpy as np 
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # define: encoder
        self.encoder = nn.Sequential(
          nn.Conv2d(3, 8, 3, 2, 1),
          nn.Conv2d(8, 16, 3, 2, 1),
        )

        # define: decoder
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(16, 8, 2, 2),
          nn.ConvTranspose2d(8, 3, 2, 2),
          nn.Tanh(),
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Total AE: return latent & reconstruct
        return encoded, decoded

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()

    autoencoder = Autoencoder()
        
    trainX = np.load(sys.argv[1])
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
    trainX = torch.Tensor(trainX)

    if use_gpu:
        autoencoder.cuda()
        trainX = trainX.cuda()

    train_dataloader = DataLoader(trainX, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(trainX, batch_size=32, shuffle=False)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    for epoch in range(20):

        cumulate_loss = 0
        for x in train_dataloader:
                
            latent, reconstruct = autoencoder(x)
            loss = criterion(reconstruct, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            cumulate_loss = loss.item() * x.shape[0]
    
    latents = []
    reconstructs = []
    for x in test_dataloader:

        latent, reconstruct = autoencoder(x)
        s = latent.shape
        latents.append(latent.cpu().detach().numpy())
        reconstructs.append(reconstruct.cpu().detach().numpy())

    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)

    latents = PCA(n_components=32, whiten=True, svd_solver='full').fit_transform(latents)
    result = KMeans(n_clusters = 2).fit(latents).labels_

    if np.sum(result[:5]) >= 3:
        result = 1 - result

    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(sys.argv[2],index=False)