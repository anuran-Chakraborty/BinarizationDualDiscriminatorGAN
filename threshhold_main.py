import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_sauvola)
import os
import numpy as np 
import cv2

input_dir = "dibco2016-dataset/"
output_dir = "Output-full/"

window_size = 25


for filename in sorted(os.listdir(input_dir)):
	
	image_gen_ = Image.open(os.path.join(input_dir, filename)).convert('L')

	image_gen = np.array(image_gen_)

	ret2,th_gen = cv2.threshold(image_gen,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	th_gen = Image.fromarray(th_gen, 'L')

	th_gen.save(output_dir+"/"+filename)

	print(filename)