from PIL import Image, ImageOps
import os

# Program to pad the images
root_dir='Data/RawData'
output_dir='Data/PaddedData'

for year in sorted(os.listdir(root_dir)):

	print(year)

	directory_data = os.path.join(root_dir, year, 'Images')
	directory_gt = os.path.join(root_dir, year, 'GT')

	for files in sorted(os.listdir(directory_data)):

		# Read the image
		img_data = Image.open(os.path.join(directory_data, files))
		img_gt = Image.open(os.path.join(directory_gt, files)).convert('L')

		# Pad the Image
		# pad_data=ImageOps.expand(img_data,border=128,fill='white')
		# pad_gt=ImageOps.expand(img_gt,border=128,fill='white')

		# Set the directories
		op_data_dir=os.path.join(output_dir, year, 'Images')
		op_gt_dir=os.path.join(output_dir, year, 'GT')

		# Create the directories if not exists
		if not os.path.exists(op_data_dir):
			os.makedirs(op_data_dir)
		if not os.path.exists(op_gt_dir):
			os.makedirs(op_gt_dir)

		# Write back the image
		img_data.save(os.path.join(op_data_dir,files))
		img_gt.save(os.path.join(op_gt_dir,files))