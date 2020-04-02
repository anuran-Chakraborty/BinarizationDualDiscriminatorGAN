import cv2
import os
from skimage import io as io
from skimage.filters import *
import matplotlib.pyplot as plt

year_test=2014
thresh = 200
real_path = './Data/Split'+str(year_test)+'/test/Images'
gen_path = './Data/Split'+str(year_test)+'/test/Output'
output_dir = './'+str(year_test)+'/PixThresh'+str(thresh)
print(year_test)

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

for fname in os.listdir(real_path):
	print(fname)   
	real_img = cv2.imread(os.path.join(real_path,fname),0) 
	gen_img = cv2.imread(os.path.join(gen_path,fname),0)

	# Crop the image
	print(gen_img.shape)

	for row in range(gen_img.shape[0]):
		for col in range(gen_img.shape[1]):
			# now threshold
			# print(real_img[row][col],gen_img[row][col])
			if(thresh<=gen_img[row][col]):
				gen_img[row][col]=255
			else:
				gen_img[row][col]=0

	cv2.imwrite(os.path.join(output_dir,fname),gen_img) 

# year_test=2013
# real_path = './Data/Split'+str(year_test)+'/test/Images'
# gen_path = './Outputs-BCELoss/Output'
# output_dir = './PixThresh'

# fname='1.bmp'
# img=io.imread(os.path.join(real_path,fname),as_gray=True)
# img=img[128:-128, 128:-128]
# # fig, ax=try_all_threshold(img, figsize=(10, 10), verbose=True)
# # fig = threshold_sauvola(img, window_size=15, k=0.2, r=None)
# fig = threshold_niblack(img, window_size=127, k=0.2)

# for row in range(len(img)):
# 	for col in range(len(img[row])):
# 		# now threshold
# 		# print(real_img[row][col],gen_img[row][col])
# 		if(fig[row][col]<img[row][col]):
# 			img[row][col]=255
# 		else:
# 			img[row][col]=0

# im=plt.imshow(img,cmap='gray', vmin=0, vmax=255)
# plt.show()
# io.imsave(os.path.join(output_dir,fname),img)