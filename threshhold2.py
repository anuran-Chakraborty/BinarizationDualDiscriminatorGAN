import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_sauvola)
import os
import numpy as np 
import cv2

year = 2014
orig_dir = "Data/Split"+str(year)+"/test/Images"
gt_dir = "Data/Split"+str(year)+"/test/GT"
input_dir = "Data/Split"+str(year)+"/test/Output"
output_gen_dir = "Data/Split"+str(year)+"/test/OutputThresh"
output_dir = "Data/Split"+str(year)+"/test/OutputOrigThresh"
all_dir = "Data/Split"+str(year)+"/test/All"

window_size = 25

# Create the directories if not exists
if not os.path.exists(output_gen_dir):
	os.makedirs(output_gen_dir)
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
if not os.path.exists(all_dir):
	os.makedirs(all_dir)

for filename in sorted(os.listdir(input_dir)):
	
	image_GT = Image.open(os.path.join(gt_dir, filename)).convert('L')
	image_orig_ = Image.open(os.path.join(orig_dir, filename)).convert('L')
	image_gen_ = Image.open(os.path.join(input_dir, filename)).convert('L')

	width, ht  = image_GT.size
	box = (128, 128, width-128, ht-128)

	# image_GT = image_GT.crop(box)
	# image_orig_ = image_orig_.crop(box)
	# image_gen_ = image_gen_.crop(box)

	# image_GT = np.array(image_GT)
	image_orig = np.array(image_orig_)
	image_gen = np.array(image_gen_)

	ret2,th_orig = cv2.threshold(image_orig,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	ret2,th_gen = cv2.threshold(image_gen,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


	th_orig = Image.fromarray(th_orig, 'L')
	th_gen = Image.fromarray(th_gen, 'L')

	imgwidth, imgheight = image_GT.size
	new_im = Image.new('L', (5*(imgwidth+5),imgheight)) #creates a new empty image, RGB mode, and size 444 by 95
	

	list_im = [image_orig_,image_GT,image_gen_,th_gen,th_orig]
	i=0
	for elem in list_im:
	    new_im.paste(elem, (i,0))
	    i+=imgwidth+5

	new_im.save(all_dir+"/"+filename)    
	th_gen.save(output_gen_dir+"/"+filename)
	th_orig.save(output_dir+"/"+filename)

	print(filename)