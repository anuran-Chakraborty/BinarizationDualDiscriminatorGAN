from PIL import Image
import os

# Program to pad the images
root_dir='Data/PaddedData'
output_dir='Data/Split'
k=1

for year1 in sorted(os.listdir(root_dir)):
	op_dir_train=os.path.join(output_dir+str(year1),'train')
	op_dir_test=os.path.join(output_dir+str(year1),'test')

	for year2 in sorted(os.listdir(root_dir)):
		print(year1,year2)

		directory_data = os.path.join(root_dir, year2, 'Images')
		directory_gt = os.path.join(root_dir, year2, 'GT')

		for files in sorted(os.listdir(directory_data)):

			# Read the image
			img_data = Image.open(os.path.join(directory_data, files))
			img_gt = Image.open(os.path.join(directory_gt, files))

			#If year1 and year2 are same then the images of year2 goes to test
			if(year1==year2):
				op_data_dir=os.path.join(op_dir_test, 'Images')
				op_gt_dir=os.path.join(op_dir_test, 'GT')

				# Create the directories if not exists
				if not os.path.exists(op_data_dir):
					os.makedirs(op_data_dir)
				if not os.path.exists(op_gt_dir):
					os.makedirs(op_gt_dir)

				img_data.save(os.path.join(op_data_dir,files))
				img_gt.save(os.path.join(op_gt_dir,files))

			else:
				op_data_dir=os.path.join(op_dir_train, 'Images')
				op_gt_dir=os.path.join(op_dir_train, 'GT')

				# Create the directories if not exists
				if not os.path.exists(op_data_dir):
					os.makedirs(op_data_dir)
				if not os.path.exists(op_gt_dir):
					os.makedirs(op_gt_dir)

				#Crop the images
				imgwidth, imgheight = img_data.size

				height = 256
				width  = 256

				step_height = 128
				step_width = 128
			
				i = 0
				j = 0

				while i+width <= imgwidth:
					j = 0
					while j+height <= imgheight:
						box = (i, j, i+width, j+height)
						cropped_img_data = img_data.crop(box)
						cropped_img_gt = img_gt.crop(box)

						cropped_img_data.save(os.path.join(op_data_dir,str(k)+'.bmp'))
						cropped_img_gt.save(os.path.join(op_gt_dir,str(k)+'.bmp'))
						
						k+=1
						j += step_height
					i += step_width