#!/usr/bin/env python3


from keras.datasets import mnist
from matplotlib import pyplot
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

device = torch.device("cpu")
class SamplingLayer(nn.Module):
	'''
	Custom layer used to generate a sample from the latent space
	'''
    def __init__(self, batch_size,latent_dim,z_mean,z_log_var):
        super(SamplingLayer, self).__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.z_mean = z_mean
        self.z_log_var = z_log_var

    def forward(self):
    	'''
		generates a sample from the latent space
		'''
		epsilon = torch.normal(0,1,size=(self.batchSize,self.latent_dim))
		return self.z_mean + torch.exp(self.z_log_var/2)*epsilon



class Encoder(nn.Module):
    def __init__(self,original_dim,latent_dim,intermediate_dim):
      super(Encoder, self).__init__()

      self.fc1 = nn.Linear(original_dim, intermediate_dim)
      self.l_relu = nn.LeakyReLU()
      self.z_mean = nn.Linear(intermediate_dim, latent_dim)
      self.z_log_var = nn.Linear(intermediate_dim, latent_dim)
      self.z = SamplingLayer(self.z_mean,self.z_log_var)

    def forward(self,x):
    	'''
    	'''

    	x = self.fc1(x)
    	x = self.l_relu(x)
    	z_mean = self.z_mean(x)
    	z_long_var = self.z_log_var(x)
    	return self.z(z_mean,z_log_var)

    
class Decoder(nn.Module):
	def __init__(self,original_dim,latent_dim,intermediate_dim):
		self.fc1 = nn.Linear(latent_dim,intermediate_dim)
		self.fc2 = nn.Linear(intermediate_dim,original_dim)
		self.l_relu = nn.LeakyReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self,x)
		x = self.fc1(x)
		x = self.l_relu(x)
		x = self.fc2(x)
		return self.sigmoid(x)
		
class AutoEncoder(nn.Module):
	def __init__(self,original_dim,latent_dim,intermediate_dim):
		self.Encoder(original_dim,latent_dim,intermediate_dim):
		self.Decoder(original_dim,latent_dim,intermediate_dim):

	def forward(self,x):
		'''
		'''
		z = self.Encoder(x)
		return self.Decoder(z)

class VaeLoss(nn.Module):
	'''
	'''
	def __init__(self):
		super(VaeLoss,self).__init__()
		self.BCELoss = nn.BCELoss()

	def forward(self,x,x_decoded_mean,z_log_var,z_mean,original_dim=28*28):
		xent_loss = original_dim*self.BCELoss(x,x_decoded_mean)
		kl_loss = -0.5*torch.sum(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
		return xent_loss + kl_loss

def main():

	(train_X, train_y), (test_X, test_y) = mnist.load_data()

	print('X_train: ' + str(train_X.shape))
	print('Y_train: ' + str(train_y.shape))
	print('X_test:  '  + str(test_X.shape))
	print('Y_test:  '  + str(test_y.shape))

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

	batchSize = 100
	original_dim = 28*28
	latent_dim = 2
	intermediate_dim = 256
	epochs = 5
	epsilonStd = 1.0

	for epoch in range(epochs):

if __name__ == "__main__":
    main()  
