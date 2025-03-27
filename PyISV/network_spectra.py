import numpy as np
import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss,self).__init__()

    def forward(self,x,y):
        criterion =  nn.MSELoss()
        loss = criterion(x, y)
        return loss
    
class Autoencoder(nn.Module):
    
    def __init__(self,embed_dim,flat_dim,kernel_size=5): 
        super(Autoencoder, self).__init__()
        
        self.kernel_size=5 

        self.flat_dim=flat_dim

        self.embed_dim=embed_dim  
            
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=self.kernel_size, padding="same"),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=self.kernel_size, padding="same"),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, padding="same"),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, padding="same"),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, padding="same"),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, padding="same"),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, padding="same"),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        #self.maxpooling_2 = nn.MaxPool1d(kernel_size=2,stride=2)
        self.embed_linear= nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim*64,self.embed_dim)
        )
        self.decode_linear = nn.Sequential(
            nn.Linear(self.embed_dim,64*self.flat_dim),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=self.kernel_size, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=self.kernel_size, padding=1)    
        )
 

        
    def forward(self, x):
        
        x = self.encoder(x)
        #print("Flat dim should be:",x.shape[2])
        embedding = self.embed_linear(x)
        z = self.decode_linear(embedding)
        z = torch.reshape(z,(z.shape[0],64,self.flat_dim))
        z = self.decoder(z)
        return  z,embedding
    
    def encode(self, x):
        
        x = self.encoder(x)
        embedding = self.embed_linear(x)
        
        return embedding 
    def decode(self, embedding):
        
        z = self.encoder(embedding)
        z = self.decode_linear(embedding)
        z = torch.reshape(z,(z.shape[0],64,self.flat_dim))
        z = self.decoder(z)
        
        return z
