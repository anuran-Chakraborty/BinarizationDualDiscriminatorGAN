import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

class LocalDiscriminator(nn.Module):

	def __init__(self):
		super(LocalDiscriminator,self).__init__()
		
		self.cnn0 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3,stride=1, padding=1)
		self.batchnorm0 = nn.BatchNorm2d(4)        #Batch normalization
		self.relu = nn.LeakyReLU()                 #RELU Activation
		#self.maxpool0 = nn.MaxPool2d(kernel_size=2) #Maxpooling reduces the size by kernel size. 256/2 = 128

		self.cnn1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3,stride=1, padding=1)
		self.batchnorm1 = nn.BatchNorm2d(8)        #Batch normalization

		#self.maxpool1 = nn.MaxPool2d(kernel_size=2) #Maxpooling reduces the size by kernel size. 128/2 = 64

		self.cnn2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,stride=1, padding=1)
		self.batchnorm2 = nn.BatchNorm2d(8)        #Batch normalization

		#self.maxpool2 = nn.MaxPool2d(kernel_size=2)   #Maxpooling reduces the size by kernel size. 64/2 = 32
		
		self.cnn3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
		self.batchnorm3 = nn.BatchNorm2d(16)
		self.maxpool3 = nn.MaxPool2d(kernel_size=2)    #Size now is 32/2 = 16
		
		#Flatten the feature maps. You have 32 feature mapsfrom cnn2. Each of the feature is of size 16x16 --> 32*16*16 = 8192
		self.fc1 = nn.Linear(in_features=16*32*32, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
		self.droput = nn.Dropout(p=0.5)                    #Dropout used to reduce overfitting
		self.fc2 = nn.Linear(in_features=4000, out_features=500)
		self.droput = nn.Dropout(p=0.5)
		self.fc3 = nn.Linear(in_features=500, out_features=50)
		self.droput = nn.Dropout(p=0.5)
		self.fc4 = nn.Linear(in_features=50, out_features=1)
		self.sigmoid = nn.Sigmoid()
		
	def forward(self,x):

		out = self.cnn0(x)
		out = self.batchnorm0(out)
		out = self.relu(out)
		#out = self.maxpool0(out)

		out = self.cnn1(out)
		out = self.batchnorm1(out)
		out = self.relu(out)
		#out = self.maxpool1(out)

		out = self.cnn2(out)
		out = self.batchnorm2(out)
		out = self.relu(out)
		#out = self.maxpool2(out)

		out = self.cnn3(out)
		out = self.batchnorm3(out)
		out = self.relu(out)
		#out = self.maxpool3(out)

		# print('Local',out.shape)

		#Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
		out = out.view(-1,16*32*32)   #-1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
		#Then we forward through our fully connected layer 
		out = self.fc1(out)
		out = self.relu(out)
		out = self.droput(out)

		out = self.fc2(out)
		out = self.relu(out)
		out = self.droput(out)
		
		out = self.fc3(out)
		out = self.relu(out)
		out = self.droput(out)
		
		out = self.fc4(out)
		out = self.sigmoid(out) 

		return out

class GlobalDiscriminator(nn.Module):

	def __init__(self):
		super(GlobalDiscriminator,self).__init__()
		
		self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,stride=1, padding=1)
		self.batchnorm1 = nn.BatchNorm2d(8)        #Batch normalization
		self.relu = nn.LeakyReLU()                 #RELU Activation
		# self.maxpool1 = nn.MaxPool2d(kernel_size=2) #Maxpooling reduces the size by kernel size. 16/2 = 8

		self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,stride=1, padding=1)
		self.batchnorm2 = nn.BatchNorm2d(16)        #Batch normalization

		self.maxpool2 = nn.MaxPool2d(kernel_size=2)   #Maxpooling reduces the size by kernel size. 16/2 = 8
				
		#Flatten the feature maps. You have 32 feature mapsfrom cnn2. Each of the feature is of size 8x8 --> 8*8*16 = 1024
		self.fc1 = nn.Linear(in_features=16*128*128, out_features=500)   #Flattened image is fed into linear NN and reduced to half size
		self.droput = nn.Dropout(p=0.5)                    #Dropout used to reduce overfitting
		self.fc2 = nn.Linear(in_features=500, out_features=50)
		self.droput = nn.Dropout(p=0.5)
		self.fc3 = nn.Linear(in_features=50, out_features=1)
		self.sigmoid = nn.Sigmoid()
		
	def forward(self,x):
		out = self.cnn1(x)
		out = self.batchnorm1(out)
		out = self.relu(out)
		
		out = self.cnn2(out)
		out = self.batchnorm2(out)
		out = self.relu(out)
		out = self.maxpool2(out)

		# print('Global',out.shape)

		#Flattening is done here with .view() -> (batch_size, 16*8*8) = (100, 1024)
		out = out.view(-1,16*128*128)   #-1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
		#Then we forward through our fully connected layer 
		out = self.fc1(out)
		out = self.relu(out)
		out = self.droput(out)

		out = self.fc2(out)
		out = self.relu(out)
		out = self.droput(out)
		
		out = self.fc3(out)
		out = self.sigmoid(out)

		return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss