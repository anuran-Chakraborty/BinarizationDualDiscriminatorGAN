import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_sauvola)
import os
import numpy as np 
import cv2

orig_dir = "Data_Dir/test/Dataset"
gt_dir = "Data_Dir/test/GT"
input_dir = "Data_Dir/test/Generated"
output_dir = "Data_Dir/test/Output"

window_size = 25


for filename in sorted(os.listdir(input_dir)):
	

	image_GT = Image.open(os.path.join(gt_dir, filename)).convert('L')
	image_orig_ = Image.open(os.path.join(orig_dir, filename)).convert('L')
	image_gen_ = Image.open(os.path.join(input_dir, filename)).convert('L')



	# image_GT = np.array(image_GT)
	image_orig = np.array(image_orig_)
	image_gen = np.array(image_gen_)

	ret2,th_orig = cv2.threshold(image_orig,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	ret2,th_gen = cv2.threshold(image_gen,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,2)
	# th4 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)

	# binary_global = image > threshold_otsu(image)
	# print(image)
	
	# thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
	# thresh_sauvola = threshold_sauvola(image, window_size=window_size)

	# binary_niblack = image > thresh_niblack
	# binary_sauvola = image > thresh_sauvola

	# binary_global = Image.fromarray(binary_global, 'L')
	# binary_niblack = Image.fromarray(binary_niblack, 'L')
	# binary_sauvola = Image.fromarray(binary_sauvola, 'L')



	th_orig = Image.fromarray(th_orig, 'L')
	th_gen = Image.fromarray(th_gen, 'L')

	imgwidth, imgheight = image_GT.size
	new_im = Image.new('L', (5*(imgwidth+5),imgheight)) #creates a new empty image, RGB mode, and size 444 by 95
	

	list_im = [image_orig_,image_GT,image_gen_,th_gen,th_orig]
	i=0
	for elem in list_im:
	    new_im.paste(elem, (i,0))
	    i+=imgwidth+5



	new_im.save(output_dir+"/"+filename)

	# th3 = Image.fromarray(th3, 'L')
	# th4 = Image.fromarray(th4, 'L')
	# new_im.paste(im, (i,0))
	# dire=output_dir+"/"+filename[:-4]
	# th2.save(dire+"_otsu.png")
	# th3.save(dire+'_mean.png')
	# th4.save(dire+'_gaussian.png')
	# binary_global.save(output_dir+"/"+filename[:-4]+"_global.png")
	# binary_niblack.save(output_dir+"/"+filename[:-4]+"_niblack.png")
	# binary_sauvola.save(output_dir+"/"+filename[:-4]+"_sauvola.png")

	print(filename)