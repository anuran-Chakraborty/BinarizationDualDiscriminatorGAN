import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import sys

from unet_model import UNet
from network import LocalDiscriminator
from network import GlobalDiscriminator
from network import FocalLoss

class ImgDataset(Dataset):

	def __init__(self, root_dir, transform_data=None, transform_gt=None):
		
		data_dir = root_dir+"/Images"
		gt_dir	 = root_dir+"/GT"

		self.img_pairs=[]

		for filename in sorted(os.listdir(data_dir)):

			img = (os.path.join(data_dir, filename))
			gt = (os.path.join(gt_dir, filename))
			
			self.img_pairs.append( (img,gt) )

		self.transform_data = transform_data
		self.transform_gt = transform_gt

	def __len__(self):

		return len(self.img_pairs)

	def __getitem__(self, index):
		
		img_f, gt_f = self.img_pairs[index]
		img=Image.open(img_f)	
		gt=Image.open(gt_f).convert('L')
		img = self.transform_data(img)	
		gt = self.transform_gt(gt)

		return img, gt

def load_images(data_path):

	global bs

	# Define the transforms to be done
	transformations_data=transforms.Compose([transforms.ToTensor()])

	transformations_gt=transforms.Compose([transforms.ToTensor()])
	
	train_dataset = ImgDataset(
	  root_dir=data_path,
	  transform_data=transformations_data,
	  transform_gt=transformations_gt)


	train_loader = torch.utils.data.DataLoader(
	  train_dataset,
	  batch_size=bs,
	  num_workers=0,
	  shuffle=True)
  
	return train_loader

def ones_target(size):
	'''
	Tensor containing ones, with shape = size
	'''
	data = Variable(torch.ones(size, 1))
	return data.to(device)

def zeros_target(size):
	'''
	Tensor containing zeros, with shape = size
	'''
	data = Variable(torch.zeros(size, 1))
	return data.to(device)	  


def split_img(img):

	split_images = []
	imgwidth, imgheight = img.size
	
	split_height = imgheight/8
	split_width  = imgwidth/8
	to_img_transform = transforms.ToTensor()
	i = 0
	j = 0

	while i+split_width < imgwidth:
		
		j = 0
		while j+split_height < imgheight:

			box = (i, j, i+split_width, j+split_height)
			split_img = img.crop(box)
			split_img = to_img_transform(split_img).to(device)

			split_images.append(split_img)
			
			j += split_height
		i += split_width	

	# split_images = transforms.ToTensor()(split_images).to(device)	
	return split_images	# Returns a list of tensors


def get_split_images(data):

	images = []
	to_img_transform = transforms.ToPILImage()
	for i in range(data.shape[0]):

		img = data[i]
		img = transforms.ToPILImage()(img.detach().cpu())
		images=images+split_img(img)
	images=torch.stack(images)

	return images	
		
def train_generator(fake_data, gt_data):
	
	no_fake_data = fake_data.size(0)    # Reset gradients
	gen_optimizer.zero_grad()    # Sample noise and generate fake data
	
	global_predictions = discriminator_g(fake_data)    # Calculate error and backpropagate
	
	split_image_list = get_split_images(fake_data)
	no_split_images = split_image_list.size(0)
	local_predictions = discriminator_l(split_image_list)

	global_prediction_loss = lossdis(global_predictions, ones_target(no_fake_data)) 
	local_prediction_loss  = lossdis(local_predictions , ones_target(no_split_images)) / (no_split_images/no_fake_data)

	# generator_level_loss   = lossgen(gt_data.view(no_fake_data, img_height*img_width),
	# 						    fake_data.view(no_fake_data, img_height*img_width)) / (img_width*img_width)
	generator_level_loss   = lossgen(fake_data,gt_data) 

	tot_error = 0.5*(global_prediction_loss + 5*local_prediction_loss) + lamda*generator_level_loss
	tot_error.backward()    # Update weights with gradients
	gen_optimizer.step()    
	return tot_error # Return error


def train_discriminator( real_data, fake_data):
	
	no_fake_data = fake_data.size(0)    # Reset gradients
	optimizer_g.zero_grad()
	optimizer_l.zero_grad()    # Sample noise and generate fake data
	
	global_predictions = discriminator_g(fake_data)    # Calculate error and backpropagate
	split_image_list = get_split_images(fake_data)
	no_split_images = split_image_list.size(0)
	local_predictions = discriminator_l(split_image_list)

	global_prediction_loss = lossdis(global_predictions, zeros_target(no_fake_data)) 
	local_prediction_loss  = lossdis(local_predictions , zeros_target(no_split_images)) / (no_split_images/no_fake_data)

	tot_fake_error = (global_prediction_loss + 5*local_prediction_loss)
	tot_fake_error.backward()    # Update weights with gradients
	 
	no_real_data = real_data.size(0)    # Reset gradients

	global_predictions = discriminator_g(real_data)    # Calculate error and backpropagate
	
	split_image_list = get_split_images(real_data)
	no_split_images = split_image_list.size(0)
	local_predictions = discriminator_l(split_image_list)

	global_prediction_loss = lossdis(global_predictions, ones_target(no_real_data)) 
	local_prediction_loss  = lossdis(local_predictions , ones_target(no_split_images)) / (no_split_images/no_real_data)

	tot_real_error = (global_prediction_loss + 5*local_prediction_loss)
	tot_real_error.backward()    # Update weights with gradients

	optimizer_g.step()
	optimizer_l.step()
	
	# Return error
	return tot_real_error+tot_fake_error


year_test=2018
data_path = './Data/Split'+str(year_test)+'/train/'
save_dir = './Output/'+str(year_test)+'/'
model_path='./models/'+str(year_test)+'/'
bs = 6

# Create the directories if not exists
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
if not os.path.exists(model_path):
	os.makedirs(model_path)

generator = UNet(n_channels=3, out_channels=1)
discriminator_g = GlobalDiscriminator()
discriminator_l = LocalDiscriminator()

resume=False
if(len(sys.argv)>1 and sys.argv[1]=='resume'):
	resume=True
	
# Load model if available
if(resume==True):
	print('Resuming training....')
	generator.load_state_dict(torch.load(os.path.join(model_path,'model_gen_latest')))
	discriminator_g.load_state_dict(torch.load(os.path.join(model_path,'model_gdis_latest')))
	discriminator_l.load_state_dict(torch.load(os.path.join(model_path,'model_ldis_latest')))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator_g = discriminator_g.to(device)
discriminator_l = discriminator_l.to(device)

optimizer_g = optim.Adam(discriminator_g.parameters(), lr=0.00005)
optimizer_l = optim.Adam(discriminator_l.parameters(), lr=0.00005)
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

lossdis = nn.BCELoss()
lossgen = FocalLoss()
lamda = 75

data_loader = load_images(data_path)
num_epochs = 2000


for epoch in range(num_epochs):
	print()
	for n_batch, (real_data, gt_data) in enumerate(data_loader):
		
		# 1. Train Discriminator
		N = real_data.size(0)
		real_data = (real_data).to(device)        
		gt_data   = (gt_data).to(device)

		# Generate fake data and detach 
		# (so gradients are not calculated for generator)
		fake_data = generator(real_data).to(device)

		# # Threshold the fake data
		# zeros=torch.zeros(fake_data.shape)
		# ones=torch.ones(fake_data.shape)

		# fake_data=torch.where(fake_data<0.6,zeros,ones).to(device) 
		# # print(fake_data)

		for i in range(1):     
			d_error= train_discriminator(gt_data, fake_data)   # Train D


		for i in range(1):
			# 2. Train Generator        # Generate fake data
			fake_data = generator(real_data).to(device)       # Train G

			# # Threshold the fake data
			# zeros=torch.zeros(fake_data.shape)
			# ones=torch.ones(fake_data.shape)

			# fake_data=torch.where(fake_data<0.6,zeros,ones).to(device) 
			# # print(fake_data)
			g_error = train_generator(fake_data, gt_data)        # Log batch error
		
		print('\rEpoch: ',epoch,'Batch',n_batch,'Gen Loss:',g_error.item(),'Dis Loss:',d_error.item())
		if (n_batch) % 50 == 0: 
			test_images = generator(real_data)
			test_images = test_images.data            
			
			grid_img = make_grid(test_images.cpu().detach(), nrow=3)
					  # Display the color image
			# plt.imshow(grid_img.permute(1, 2, 0))
			save_image(real_data.cpu().detach(),save_dir+str(epoch)+'_image_'+str(n_batch)+'_real.png', nrow=4)
			save_image(test_images.cpu().detach(),save_dir+str(epoch)+'_image_'+str(n_batch)+'_gen.png', nrow=4)
			save_image(gt_data.cpu().detach(),save_dir+str(epoch)+'_image_'+str(n_batch)+'_gt.png', nrow=4)

			# Save the latest models to resume training
			torch.save(generator.state_dict(), os.path.join(model_path,'model_gen_latest'))
			torch.save(discriminator_g.state_dict(), os.path.join(model_path,'model_gdis_latest'))
			torch.save(discriminator_l.state_dict(), os.path.join(model_path,'model_ldis_latest'))

	# Save every model of generator		
	torch.save(generator.state_dict(), os.path.join(model_path,'model_'+str(epoch)))
