# set the matplotlib backend so figures can be saved in the background
from posixpath import split
import matplotlib
matplotlib.use("Agg")
import csv
# import the necessary packages
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import model_from_json

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import Contours 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
set_session(sess)
import common


def alphaBlend( img1, img2, mask):
		if mask.ndim==3 and mask.shape[-1] == 3:
			alpha = mask/255.0
		else:
			alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
		blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha )
		return blended

def resize_image( image,max_w=128, max_h=128):
	height,width= image.shape[:2]
	scale_1 = float(max_w/ width)
	scale_2 = float(max_h/ height)
	scale = min(scale_1, scale_2)
	width= int(width*scale)
	height=int(height*scale)
	image= cv2.resize(image,(width,height))
	return image

def get_background(image , mask):
	h, w = image.shape[:2]
	gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
	m = cv2.mean(gray, mask)
	_, th = cv2.threshold(gray,m[0],255,cv2.THRESH_BINARY_INV)
	th = cv2.bitwise_and(mask, th)
	m = cv2.mean(image, th)
	output = np.zeros((h,w,3), image.dtype)
	output[:, :] = (1*m[0], 1*m[1] ,1* m[2])
	m = cv2.mean(output)
	kernel = np.ones((3, 3), np.uint8)
	mask = cv2.erode(mask, kernel) 
	mask = cv2.GaussianBlur(mask, (5,5), 0)
	blended = alphaBlend( output, image, mask)
	# blended = cv2.GaussianBlur(blended, (3,3), 0)
	# cv2.imwrite("blended.jpg" , th)
   
	return blended

def create_mask(image, est=0.3):
	h, w = image.shape[:2]
	mask = 255*np.ones((h,w),np.uint8)
	ex = 0
	ey = 0
	size = max(w, h)
	if h > w:
		ex = h-w
	elif w> h:
		ey = w- h
	ex += int(est*size)
	ey += int(est*size)
	mask = cv2.copyMakeBorder(mask, ey//2, ey//2, ex//2, ex//2, cv2.BORDER_CONSTANT, value=(0, 0 , 0))
	image = cv2.copyMakeBorder(image, ey//2, ey//2, ex//2, ex//2, cv2.BORDER_CONSTANT, value=(0, 0 , 0))
	# mask = cv2.rectangle(mask, (0,0), (w, h), (0), 1)
	return image, mask

class SerialDetection():
	def __init__(self):
		self.num_classes = 36
		# input image dimensions
		self.img_rows, self.img_cols = 28, 28
		json_file = open('models/model_0616_crop.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		self.loaded_model.load_weights("models/model_weights_0616_crop.h5")
		print("Loaded model from disk")
		self.loaded_model.summary()
		self.img_debug = None
		with open('model_name.txt','r') as f:
			labels = [line.strip('\n') for line in f.readlines()]
		self.labels = labels[0].split(",")
	def predict(self, image, pose=-1):
		image = cv2.bitwise_not(image)
		# image = common.preprocess(image)
		image = resize_image(image)
		image, mask = create_mask(image)
		
		image = get_background(image, mask)
		gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
		gray = cv2.resize(gray, (self.img_rows, self.img_cols))
		
		gray = img_to_array(gray)
		gray = np.array(gray, dtype="float") / 255.0
		gray = np.expand_dims(gray, axis=0)
		prediction = self.loaded_model.predict(gray)[0]
		# print("prediction " , prediction, image.shape)
		bestclass = ''
		bestconf = -1
		idx_cls = -1
		for n in range(self.num_classes):
			if pose > 1 and n > 9 :
				break
			if (prediction[n] > bestconf):
				idx_cls = n
				bestclass = self.labels[n]
				bestconf = prediction[n]
		return idx_cls, bestclass, bestconf
	
	def getSerialForm(self, image):
		img_serial , listChar = Contours.splitCharFromForm(image)
		# print("listChar" , len(listChar) , listChar)
		serial_number = "" 
		for i, box in enumerate(listChar):
			xmin = box[0][0]
			ymin = box[0][1]
			xmax = box[1][0]
			ymax = box[1][1]
			im_char = img_serial[ymin:ymax, xmin:xmax]
			idx_cls, bestclass, bestconf = model.predict(im_char , pose= i +1 )
			serial_number += bestclass
			# cv2.rectangle(img_serial,(xmin, ymin),(xmax, ymax),(255,0,0),1)
		
		return img_serial , serial_number
		
if __name__ == '__main__':
	model = SerialDetection()
	imagePaths = sorted(list(paths.list_images("/media/anlab/ssd_samsung_256/dungtd/SplitHandWriting/TestImage/input")))
	folder_save = "results"
	if not os.path.exists(folder_save):
		os.mkdir(folder_save)
	for imagePath in imagePaths:
		print("path " ,  imagePath)
		basename = os.path.basename(imagePath)
		image = cv2.imread(imagePath)
		img_serial , serial_number = model.getSerialForm(image)
		
		path_out =os.path.join(folder_save , f'{serial_number}_{basename}')
		cv2.imwrite(path_out, img_serial)
		exit()
 

 
	# imagePaths = sorted(list(paths.list_images("/media/anlab/ssd_samsung_256/dungtd/EMNIST/source/DataCrop/ResultCrop")))
	# folder_save = "results"
	# if not os.path.exists(folder_save):
	# 	os.mkdir(folder_save)
	# for imagePath in imagePaths:
	# 	image = cv2.imread(imagePath)
	# 	base_path = imagePath.split(os.path.sep)[-1]
	# 	pose = int(base_path.split("_")[0])
	# 	idx_cls, bestclass, bestconf = model.predict(image , pose= pose)
	# 	img_out = model.img_debug
	# 	path_out =os.path.join(folder_save , f'{bestclass}_{imagePath.split(os.path.sep)[-1]}') 
	
	# 	cv2.imwrite(path_out ,img_out )

