import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
import os

input_dir=os.path.join('Outputs-BCELoss','OutputThresh')

for fname in os.listdir(input_dir):
	print(fname)   
	# Reading image from folder where it is stored 
	img = cv2.imread(os.path.join(input_dir,fname)) 
	  
	# denoising of image saving it into dst image 
	dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7)

	cv2.imwrite('Denoised/'+fname,dst) 