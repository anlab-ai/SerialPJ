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
import numpy
from scipy.spatial import distance
from sklearn import preprocessing
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

def resize_image( image,input_size=128):
	height,width= image.shape[:2]
	scale_1 = float(input_size/ width)
	scale_2 = float(input_size/ height)
	scale = min(scale_1, scale_2)
	width= int(width*scale)
	height=int(height*scale)
	image= cv2.resize(image,(width,height))
	return image

def resize_image_min( image,input_size=128):
	height,width= image.shape[:2]
	scale_1 = float(input_size/ width)
	scale_2 = float(input_size/ height)
	scale = max(scale_1, scale_2)
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

def create_mask(image, est=0.2):
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

def create_mask_non_size(image, est=0.2):
	h, w = image.shape[:2]
	mask = 255*np.ones((h,w),np.uint8)
	ex = 0
	ey = 0
	size = min(w, h)
	# if h > w:
	# 	ex = h-w
	# elif w> h:
	# 	ey = w- h
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
		self.net = cv2.dnn.readNetFromTensorflow("model_compares/frozen_inference_graph_dnn.pb", "model_compares/frozen_inference_graph_dnn.pbtxt")
		self.image_size = 1280
		self.feature_template = []
		im = cv2.imread("template_data/1.png", 0)
		# im = cv2.GaussianBlur(im, (3,3), 0)
		fea = self.getFeature( im)
		self.feature_template.append(fea)
		im = cv2.imread("template_data/2.png", 0)
		# im = cv2.GaussianBlur(im, (3,3), 0)
		fea = self.getFeature(im)
		self.feature_template.append(fea)

		im = cv2.imread("template_data/3.png", 0)
		# im = cv2.GaussianBlur(im, (3,3), 0)
		fea = self.getFeature(im)
		self.feature_template.append(fea)
		im = cv2.imread("template_data/4.png", 0)
		# im = cv2.GaussianBlur(im, (3,3), 0)
		fea = self.getFeature(im)
		self.feature_template.append(fea)
		
		
	def predict(self, image, is_digit=False):
		image = cv2.bitwise_not(image)
		# image = common.preprocess(image)
		image = resize_image(image , input_size=28)
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
			if is_digit and n > 9:
				break
			if (prediction[n] > bestconf):
				idx_cls = n
				bestclass = self.labels[n]
				bestconf = prediction[n]
		return idx_cls, bestclass, bestconf

	def getFeature(self, image):
		# image = self.pre_process(image)
		blob = cv2.dnn.blobFromImage(image, 1/255.0, (105, 105))
		# blob = cv2.dnn.blobFromImage(im, 1.0, size_net, mean, )
		self.net.setInput(blob)
		outputlayers = self.net.getUnconnectedOutLayersNames()
		feature = self.net.forward( outputlayers)
		feature = np.array(feature)[0]
		feature = preprocessing.normalize(feature)
		return feature

	def pre_process(self, image):
		image = cv2.bitwise_not(image)
		
		# image = common.preprocess(image)
		image, mask = create_mask_non_size(image, est=0.2)
		
		image = get_background(image, mask)
		image = cv2.bitwise_not(image)
		
		gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
		
		return gray

	def selection(self, im_query , feature1, feature2, thresh=0.01):
		gray = self.pre_process(im_query)
		# cv2.imwrite("im.jpg" , gray)
		fea = self.getFeature( gray)
		score1 = numpy.linalg.norm(fea - feature1)
		score2 = numpy.linalg.norm(fea - feature2)
		# print("score compare " , score1 , score2)
		if score1 < score2 :
			return 1 
		else :
			return 2
		# if score1 < score2 - thresh:
		# 	return 1 
		# elif score2 < score1 - thresh:
		# 	return 2
		# else:
		# 	return 0
		

	def getSerialForm(self, image):
		img = resize_image_min(image,input_size=self.image_size )
		box_crop = [968,830, 1220, 884]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		# print("image shape", img.shape)
		listChar = Contours.splitCharFromForm(img_serial)
		# print("listChar" , len(listChar) , listChar)
		serial_number = "" 
		est = 0
		h, w = img_serial.shape[:2]

		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			
			im_char = img_serial[ymin:ymax, xmin:xmax]
			is_digit = False
			if i  > 0  :
				is_digit = True
			idx_cls, bestclass, bestconf = model.predict(im_char , is_digit )
			serial_number += bestclass
		# for i, box in enumerate(listChar):
		# 	xmin = box[0][0]
		# 	ymin = box[0][1]
		# 	xmax = box[1][0]
		# 	ymax = box[1][1]
			
			# cv2.rectangle(img_serial,(xmin, ymin),(xmax, ymax),(255,0,0),1)
		
		return img_serial , serial_number

	def checkSelection(self, image):
		index_in_out = 0
		index_electric = 0
		
		img = resize_image_min(image,input_size=self.image_size )
		scale = image.shape[0]/img.shape[0]
		box_in_out = [int(scale*950),int(scale*585),int(scale*1063),int(scale*627)]
		# print("box_in_out" , box_in_out)
		im_in_out = image[box_in_out[1]:box_in_out[3],box_in_out[0]:box_in_out[2]]
		# print("image shape", img.shape)
		# in-out detection 
		ret, box_info = Contours.getInfo(im_in_out)
		if ret:
			im_crop = im_in_out[box_info[1]:box_info[3] , box_info[0]:box_info[2]]
			# gray = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
			
			index_in_out = self.selection(im_crop, self.feature_template[0] , self.feature_template[1])
		
		# ElectricMotor detection 
		box_electric = [int(scale*1140),int(scale*588),int(scale*1235),int(scale*625)]
		# print("box_electric" , box_electric)
		im_electric = image[box_electric[1]:box_electric[3],box_electric[0]:box_electric[2]]
		ret, box_info2 = Contours.getInfo(im_electric)
		if ret:
			im_crop = im_electric[box_info2[1]:box_info2[3] , box_info2[0]:box_info2[2]]
			# gray = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
			index_electric = self.selection(im_crop, self.feature_template[2] , self.feature_template[3])
			
		# print("index_in_out " , index_in_out)
		# cv2.imwrite("output.jpg", im_in_out)
		# box_info = [box_info[0] + box_in_out[0], box_info[1] + box_in_out[1] , box_info[2] + box_in_out[0] , box_info[3] + box_in_out[1]]
		# box_info2 = [box_info2[0] + box_electric[0], box_info2[1] + box_electric[1] , box_info2[2] + box_electric[0] , box_info2[3] + box_electric[1]]
		return index_in_out , index_electric 
		
if __name__ == '__main__':
	model = SerialDetection()
	imagePaths = sorted(list(paths.list_images("/media/anlab/ssd_samsung_256/dungtd/SerialOCR/input")))
	folder_save = "results"
	if not os.path.exists(folder_save):
		os.mkdir(folder_save)
	for imagePath in imagePaths:
		print("path " ,  imagePath)
		# imagePath = "/media/anlab/ssd_samsung_256/dungtd/SerialOCR/input/MFG No.042214830 LK-47VH-02E.jpg"
		basename = os.path.basename(imagePath)
		image = cv2.imread(imagePath)
		index_in_out , index_electric   = model.checkSelection(image)
		img_serial , serial_number= model.getSerialForm(image)
		# image = resize_image_min(image,input_size=1280 )
		# cv2.rectangle(image,(box_info[0], box_info[1]),(box_info[2], box_info[3]),(255,0,0),1)
		# cv2.rectangle(image,(box_info2[0], box_info2[1]),(box_info2[2], box_info2[3]),(255,0,0),1)
		path_out =os.path.join(folder_save , f'{serial_number}_{index_in_out}_{index_electric}_{basename}')
		cv2.imwrite(path_out, image)
		# exit()
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

