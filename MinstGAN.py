#!/usr/bin/env python3


from keras.datasets import mnist
from matplotlib import pyplot
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class SamplingLayer(nn.Module):
    '''
    Custom layer used to generate a sample from the latent space
    '''
    def __init__(self, latent_dim):
        super(SamplingLayer, self).__init__()
        self.latent_dim = latent_dim

    def forward(self,z_mean,z_log_var):
        '''
        generates a sample from the latent space
        '''
        epsilon = torch.normal(0,1,size=(10,self.latent_dim))
        return z_mean + torch.exp(z_log_var/2)*epsilon



class Encoder(nn.Module):
    def __init__(self,original_dim,latent_dim,intermediate_dim):
      super(Encoder, self).__init__()

      self.fc1 = nn.Linear(original_dim, intermediate_dim)
      self.l_relu = nn.LeakyReLU()
      self.z_mean = nn.Linear(intermediate_dim, latent_dim)
      self.z_log_var = nn.Linear(intermediate_dim, latent_dim)
      self.z = SamplingLayer(latent_dim)

    def forward(self,x):
        '''
        '''

        x = self.fc1(x)
        x = self.l_relu(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return self.z.forward(z_mean,z_log_var)

    
class Decoder(nn.Module):
    def __init__(self,original_dim,latent_dim,intermediate_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim,intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim,original_dim)
        self.l_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.fc1(x)
        x = self.l_relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)
        
class AutoEncoder(nn.Module):
    def __init__(self,original_dim,latent_dim,intermediate_dim):
        super(AutoEncoder, self).__init__()
        self.Encoder = Encoder(original_dim,latent_dim,intermediate_dim)
        self.Decoder = Decoder(original_dim,latent_dim,intermediate_dim)

    def forward(self,x):
        '''
        '''
        x = self.Encoder.forward(x)
        #print(x)
        #print(x.shape)
        z_mean = x[:,0]
        #print(z_mean)
        z_log_var = x[:,-1]
        #print(z_log_var)
        x_decoded_mean = []
        for i in range(x.shape[0]):
            x_decoded_mean.append(self.Decoder.forward(x[i]))

        #print(torch.stack(x_decoded_mean))
        return torch.stack(x_decoded_mean), z_mean, z_log_var

def vaeLoss(x,x_decoded_mean,z_mean,z_log_var,original_dim=28*28):
    '''
    '''
    BCELoss = nn.BCELoss()
    
    try:
        xent_loss = original_dim*BCELoss(x_decoded_mean,x)
    except:
        print(x_decoded_mean)
        print(torch.min(x_decoded_mean))
        print(x)
        print(torch.min(x))
    kl_loss = -0.5*torch.sum(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
    return xent_loss + kl_loss

def main():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print('X_train: ' + str(x_train.shape))
    print('Y_train: ' + str(y_train.shape))
    print('X_test:  '  + str(x_test.shape))
    print('Y_test:  '  + str(y_test.shape))

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


    print('X_train: ' + str(x_train.shape))
    print('Y_train: ' + str(y_train.shape))
    print('X_test:  '  + str(x_test.shape))
    print('Y_test:  '  + str(y_test.shape))

    batchSize = 10
    original_dim = 28*28
    latent_dim = 2
    intermediate_dim = 256
    epochs = 5
    epsilonStd = 1.0

    auto_model = AutoEncoder(original_dim,latent_dim,intermediate_dim)
    opt = torch.optim.Adam(auto_model.parameters())
    for name,l in auto_model.named_children():
        print(f"{name}:{l}")

    for epoch in tqdm(range(epochs),ascii=True,desc='Epochs'):
        print('Epoch: ',epoch)
        
        for i in tqdm(range(x_train.shape[0]),ascii=True,desc='i'):
            x_decoded_mean, z_mean, z_log_var = auto_model.forward(torch.from_numpy(x_train[i]))
            #print('###################################################################')
            i_train = torch.stack([torch.tensor(x_train[i]) for j in range(batchSize)])
            
            '''
            print('i_train \n',i_train)
            print(i_train.shape)
            print('x_decoded_mean \n',x_decoded_mean)
            print(x_decoded_mean.shape)
            print('z_mean \n',z_mean)
            print(z_mean.shape)
            print('z_log_var \n',z_log_var)
            print(z_log_var.shape)
            '''
            loss = vaeLoss(i_train,x_decoded_mean,z_mean,z_log_var)
            #print(loss)
            loss /= batchSize
            #print('loss: ',loss)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('loss: ',loss)
    for i in range(10):
        out = auto_model.forward(x_test[i])
        print(out)
        pyplot.imshow(out)

if __name__ == "__main__":
    main()  
