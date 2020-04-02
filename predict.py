import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from PIL import Image
import os
import numpy as np
from PIL import Image
from unet_model import UNet

def normalize(arr):
	"""
	Linear normalization
	http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
	"""
	arr = np.array(arr)
	arr = arr.astype('float')
	# Do not touch the alpha channel
	for i in range(3):
		minval = arr[...,i].min()
		maxval = arr[...,i].max()
		if minval != maxval:
			arr[...,i] -= minval
			arr[...,i] *= (255.0/(maxval-minval))
	new_img = Image.fromarray(arr.astype('uint8'),'L')
	return new_img

year = 2014
print(year)
input_dir = "Data/Split"+str(year)+"/test/Images"
output_dir = "Data/Split"+str(year)+"/test/Output"
model_path = "models/"+str(year)+"/model_gen_latest" 

# Specify the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = UNet(n_channels=3, out_channels=1)
generator.load_state_dict(torch.load(model_path))
generator.eval()

# Create the directories if not exists
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

for filename in sorted(os.listdir(input_dir)):

	img = Image.open(os.path.join(input_dir, filename))
	# img = normalize(img)
	img = torch.stack( [transforms.ToTensor()(img)])

	output_img = generator(img)


	save_image(output_img.cpu(), output_dir+"/"+filename)
	print(filename)

