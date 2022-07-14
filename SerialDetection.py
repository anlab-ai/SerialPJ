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
from model_CRNN_3_channels import get_model
from parameters import letters

latin_hand_writting_style_jp_model = "./latin_hand_writting_style_jp_model/"
model_compares_text_options = "./model_compares_text_options/"
template_image_text_options = "./template_image_text_options/"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
set_session(sess)
import common

def convertImage(img_pred):
    
    img_pred = cv2.resize(img_pred, (256, 64))
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    return img_pred

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
	output[:, :] = (0.97*m[0], 0.97*m[1] ,0.97* m[2])
	m = cv2.mean(output)
	kernel = np.ones((3, 3), np.uint8)
	mask = cv2.erode(mask, kernel) 
	mask = cv2.GaussianBlur(mask, (5,5), 0)
	blended = alphaBlend( output, image, mask)
	# blended = cv2.GaussianBlur(blended, (3,3), 0)
	# cv2.imwrite("blended.jpg" , th)
   
	return blended

def create_mask(image, est=0.5):
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
	
	ex += int(est*size)
	ey += int(est*size)
	mask = cv2.copyMakeBorder(mask, ey//2, ey//2, ex//2, ex//2, cv2.BORDER_CONSTANT, value=(0, 0 , 0))
	image = cv2.copyMakeBorder(image, ey//2, ey//2, ex//2, ex//2, cv2.BORDER_CONSTANT, value=(0, 0 , 0))
	return image, mask

class SerialDetection():
	def __init__(self):
		self.num_classes = 36
		# input image dimensions
		self.img_rows, self.img_cols = 28, 28
		json_file = open(latin_hand_writting_style_jp_model + 'model_0707_crop.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		self.loaded_model.load_weights(latin_hand_writting_style_jp_model + "model_weights_0707_crop.h5")
		print("Loaded model from disk")
		# self.loaded_model.summary()
		self.img_debug = None
		with open(latin_hand_writting_style_jp_model + 'model_name.txt','r') as f:
			labels = [line.strip('\n') for line in f.readlines()]
		self.labels = labels[0].split(",")
		self.image_size = 1280
		weights_path = model_compares_text_options + 'model_24062022.hdf5'
		self.net = get_model( weights_path = weights_path)
		self.net.summary()
		images = []
		for i in range(2):
			im = cv2.imread(template_image_text_options + f"side_{i+1}.png", 0)
			images.append(im)
		for i in range(2):
			im = cv2.imread(template_image_text_options + f"electric_type_{i+1}.png", 0)
			images.append(im)
		for i in range(3):
			im = cv2.imread(template_image_text_options + f"contruction_{i+1}_0.png", 0)
			images.append(im)

		for i in range(4):
			im = cv2.imread(template_image_text_options + f"maker_{i+1}.png", 0)
			images.append(im)
		self.feature_template = []
		
		features= self.getFeature_update(images)
		for i in range(len(images)):
			fea = features[i]
			fea = np.sum(fea, axis=0)
			
			# fea = fea.flatten()
			fea = fea / np.sqrt(np.sum(fea**2))
			self.feature_template.append(fea)
			# exit()
		
		
	def predict_char(self, image, is_char=False):
		image = cv2.bitwise_not(image)
		# image = common.preprocess(image)
		image = resize_image(image , input_size=26)
		image, mask = create_mask(image)
		
		image = get_background(image, mask)
		gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
		gray = cv2.resize(gray, (self.img_rows, self.img_cols))
		# cv2.imwrite("output.jpg", gray)
		gray = img_to_array(gray)
		gray = np.array(gray, dtype="float") / 255.0
		gray = np.expand_dims(gray, axis=0)
		prediction = self.loaded_model.predict(gray)[0]
		# print("prediction " , prediction, image.shape)
		bestclass = ''
		bestconf = -1
		idx_cls = -1
		for n in range(self.num_classes):
			if is_char and n <= 9:
				continue
			if (prediction[n] > bestconf):
				idx_cls = n
				bestclass = self.labels[n]
				bestconf = prediction[n]
		# print("bestclass" , bestclass, is_digit)
		# exit()
		return idx_cls, bestclass, bestconf


	def predict(self, image, is_digit=False):
		image = cv2.bitwise_not(image)
		# image = common.preprocess(image)
		image = resize_image(image , input_size=26)
		image, mask = create_mask(image)
		
		image = get_background(image, mask)
		gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
		gray = cv2.resize(gray, (self.img_rows, self.img_cols))
		# cv2.imwrite("output.jpg", gray)
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
		# print("bestclass" , bestclass, is_digit)
		# exit()
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

	def getFeature_update(self, images):
		datas = []
		for im in images:
			im = convertImage(im)
			datas.append(im)
		datas = np.array(datas)
		features = self.net.predict(datas)
		return features

	def pre_process(self, image):
		image = cv2.bitwise_not(image)
		
		# image = common.preprocess(image)
		image, mask = create_mask_non_size(image, est=0.2)
		
		image = get_background(image, mask)
		image = cv2.bitwise_not(image)
		
		gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
		
		return gray

	def selection(self, im_query , features_template, thresh=0.01):
		index = 0 
		if len(features_template) < 2 :
			return index
		gray = self.pre_process(im_query)

		features = self.getFeature_update( [gray])
		fea = features[0]
		fea = np.sum(fea, axis=0)
		# fea =  fea.flatten()
		# feature1 = feature1.flatten()
		# feature2 = feature2.flatten()
		# feature1 = feature1 / np.sqrt(np.sum(feature1**2))
		# feature2 = feature2 / np.sqrt(np.sum(feature2**2))
		fea = fea / np.sqrt(np.sum(fea**2))
		scores = []
		for i, f in enumerate(features_template):
			score = numpy.linalg.norm(fea- f)
			scores.append(score)
		
		index = np.argmin(scores) + 1
		# print("scores " , scores)
		return index, scores[index-1]

	#Get input is image and return	MotorLotNo
	def getMotorLotNoForm(self, image):
		img = resize_image_min(image,input_size=self.image_size )
		scale = self.image_size/image.shape[1]
		box_crop = [int(scale*457),int(scale*806),int(scale*1099),int(scale*900)]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		# print("image shape", img.shape)
		listChar = Contours.splitCharFromForm(img_serial)
		# print("listChar" , len(listChar) , listChar)
		serial_number = "" 
		est = 2
		h, w = img_serial.shape[:2]
		# img_serialcp = img_serial.copy()
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			
			im_char = img_serial[ymin:ymax, xmin:xmax]
			
			is_char = False
			if i  == 2  :
				is_char = True
			idx_cls, bestclass, bestconf = self.predict_char(im_char , is_char )
			serial_number += bestclass
			# cv2.rectangle(img_serialcp,(xmin, ymin),(xmax, ymax),(255,0,0),1)
			# cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/'+basename,img_serialcp)
		return img_serial , serial_number

	#Get input is image and return power value
	def getPowerValue(self, image):
		img = resize_image_min(image,input_size=self.image_size )
		scale = self.image_size/image.shape[1]
		box_crop = [int(scale*1730),int(scale*1215),int(scale*1950),int(scale*1290)]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		listChar = Contours.splitCharFromForm(img_serial)
		serial_number = "" 
		est = 2
		h, w = img_serial.shape[:2]
		img_serialcp = img_serial.copy()
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			
			im_char = img_serial[ymin:ymax, xmin:xmax]
			
			is_digit = True
			# if i  != 1  :
			# 	is_digit = True
			if i == 1:
				serial_number+='.'	
			idx_cls, bestclass, bestconf = self.predict(im_char , is_digit )
			serial_number+=bestclass
			# cv2.rectangle(img_serialcp,(xmin, ymin),(xmax, ymax),(255,0,0),1)
			# cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/'+basename,img_serialcp)
		return img_serialcp , serial_number

	#Get input is image and return Dynamic Viscosity value
	def getDynamicViscosity(self, image):
		img = resize_image_min(image,input_size=self.image_size )
		scale = self.image_size/image.shape[1]
		box_crop = [int(scale*2045),int(scale*1220),int(scale*2168),int(scale*1281)]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		ret, bbox = Contours.getInfo(img_serial,SingleChar = True,areaRatio=[0.1,0.1,0.002])
		im_crop = img_serial[bbox[1]:bbox[3] , bbox[0]:bbox[2]]
		idx_cls, bestclass, bestconf = self.predict(im_crop , is_digit=True )
		# cv2.rectangle(img_serial,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(255,0,0),1)
		# cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/'+basename,img_serialcp)
		return img_serial , bestclass

	#Get input is image and return V value
	def getVValue(self, image):	
		img = resize_image_min(image,input_size=self.image_size )
		scale = self.image_size/image.shape[1]
		box_crop = [int(scale*1875),int(scale*1367),int(scale*2389),int(scale*1445)]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		imgcp = img_serial.copy()
		listChar = Contours.splitCharFromForm(img_serial,Color = [False, True])
		serial_number = "" 
		est = 2
		h, w = img_serial.shape[:2]
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			im_char = img_serial[ymin:ymax, xmin:xmax]
			is_digit = True
			idx_cls, bestclass, bestconf = self.predict(im_char , is_digit)
			serial_number += bestclass
			# cv2.rectangle(imgcp,(xmin, ymin),(xmax, ymax),(255,0,0),1)
			# cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/'+basename,img_serial)
		return imgcp , serial_number

	#Get input is image and return Hz value
	def getHzValue(self, image):	
		img = resize_image_min(image,input_size=self.image_size )
		scale = self.image_size/image.shape[1]
		box_crop = [int(scale*1875),int(scale*1450),int(scale*2389),int(scale*1523)]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		imgcp = img_serial.copy()
		listChar = Contours.splitCharFromForm(img_serial)
		serial_number = "" 
		est = 2
		h, w = img_serial.shape[:2]
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			im_char = img_serial[ymin:ymax, xmin:xmax]
			is_digit = True
			idx_cls, bestclass, bestconf = self.predict(im_char , is_digit )
			serial_number += bestclass
			# cv2.rectangle(imgcp,(xmin, ymin),(xmax, ymax),(255,0,0),1)
			# cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/'+basename,img_serial)
		return imgcp , serial_number

	#Get input is image and return min value
	def getMinValue(self, image):	
		img = resize_image_min(image,input_size=self.image_size )
		scale = self.image_size/image.shape[1]
		box_crop = [int(scale*1885),int(scale*1541),int(scale*2389),int(scale*1625)]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		imgcp = img_serial.copy()
		listChar = Contours.splitCharFromForm(img_serial)
		serial_number = "" 
		est = 2
		h, w = img_serial.shape[:2]
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			im_char = img_serial[ymin:ymax, xmin:xmax]
			is_digit = True
			idx_cls, bestclass, bestconf = self.predict(im_char , is_digit )
			serial_number += bestclass
			# cv2.rectangle(imgcp,(xmin, ymin),(xmax, ymax),(255,0,0),1)
			# cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/'+basename,img_serial)
		return imgcp , serial_number
	#Get input is image and return serial number
	def getSerialForm(self, image):
		img = resize_image_min(image,input_size=self.image_size )
		box_crop = [968,830, 1233, 887]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		# print("image shape", img.shape)
		listChar = Contours.splitCharFromForm(img_serial)
		# print("listChar" , len(listChar) , listChar)
		serial_number = "" 
		est = 2
		h, w = img_serial.shape[:2]
		
		score = {}
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			
			im_char = img_serial[ymin:ymax, xmin:xmax]
			
			is_digit = False
			if i  > 0  :
				is_digit = True
			idx_cls, bestclass, bestconf = self.predict(im_char , is_digit )
			serial_number += bestclass
			score[i] = bestconf
		# for i, box in enumerate(listChar):
		# 	xmin = box[0][0]
		# 	ymin = box[0][1]
		# 	xmax = box[1][0]
		# 	ymax = box[1][1]
			
			# cv2.rectangle(img_serial,(xmin, ymin),(xmax, ymax),(255,0,0),1)
		
		return img_serial , serial_number, score

	def checkSelection(self, image):
		index_in_out = 0
		index_electric = 0
		index_contruction = 0
		index_maker = 0
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
			index_in_out, _ = self.selection(im_crop, self.feature_template[0:2])
   
		# ElectricMotor detection 
		box_electric = [int(scale*1140),int(scale*588),int(scale*1235),int(scale*625)]
		# print("box_electric" , box_electric)
		im_electric = image[box_electric[1]:box_electric[3],box_electric[0]:box_electric[2]]
		ret, box_info2 = Contours.getInfo(im_electric)
		if ret:
			im_crop = im_electric[box_info2[1]:box_info2[3] , box_info2[0]:box_info2[2]]
			index_electric, _ = self.selection(im_crop, self.feature_template[2:4])
		
		# Contruction detection
		box_contruction = [int(scale*955),int(scale*543),int(scale*1230),int(scale*590)]
		# print("box_electric" , box_electric)
		im_contruction = image[box_contruction[1]:box_contruction[3],box_contruction[0]:box_contruction[2]]
		ret, box_info3 = Contours.getInfo(im_contruction,areaRatio=[0.005,0.01,0.02])
		if ret:
			im_crop = im_contruction[box_info3[1]:box_info3[3] , box_info3[0]:box_info3[2]]
			index_contruction, _ = self.selection(im_crop, self.feature_template[4:7])
		
  
		# maker detection
		box_maker = [int(scale*955),int(scale*473),int(scale*1230),int(scale*517)]
		# print("box_electric" , box_electric)
		im_maker = image[box_maker[1]:box_maker[3],box_maker[0]:box_maker[2]]
		ret, box_info3 = Contours.getInfo(im_maker,areaRatio=[0.003,0.01,0.01])
		if ret:
			im_crop_full = im_maker[box_info3[1]:box_info3[3] , box_info3[0]:box_info3[2]]
			h1, w1 = im_crop_full.shape[:2]
			ratio = w1/ (h1 +1)
			im_crop1 = None
			im_crop = im_crop_full
			if ratio > 3.5 :
				im_crop1 = im_crop_full[0:h1, w1//2:w1]
				im_crop = im_crop_full[0:h1, 0:w1//2]
				
			index_maker, score = self.selection(im_crop, self.feature_template[7:11])
			if im_crop1 is not None:
				index_maker2, score2 = self.selection(im_crop1, self.feature_template[7:11])
				if score2 < score:
					index_maker= index_maker2
				im_crop = im_crop1
			
	
		# print("index_in_out " , index_in_out)
		# cv2.imwrite("output.jpg", im_in_out)
		# box_info = [box_info[0] + box_in_out[0], box_info[1] + box_in_out[1] , box_info[2] + box_in_out[0] , box_info[3] + box_in_out[1]]
		# box_info3 = [box_info3[0] + box_contruction[0], box_info3[1] + box_contruction[1] , box_info3[2] + box_contruction[0] , box_info3[3] + box_contruction[1]]
  		# box_info2 = [box_info2[0] + box_electric[0], box_info2[1] + box_electric[1] , box_info2[2] + box_electric[0] , box_info2[3] + box_electric[1]]
		return index_in_out , index_electric, index_contruction, index_maker
		
if __name__ == '__main__':
	model = SerialDetection()
	imagePaths = sorted(list(paths.list_images("/home/anlab/Downloads/LK_image_from_pdf-20220620T085925Z-001/LK_image_from_pdf")))
	# imagePaths = sorted(list(paths.list_images("LK_image_from_pdf")))
	folder_save = "results"
	if not os.path.exists(folder_save):
		os.mkdir(folder_save)
	count_index_in_out = 0
	count_index_electric = 0
	count_index_contruction = 0
	count_SerialNo = 0
	count_maker = 0
 
	listCountInout = {}
	listCountElectric = {}
	listCountContruction = {}
	listSerialNo = {}
	listMaker = {}
	with open('expected_result_70_images_update.csv') as csv_file:
			csv_reader = csv_file.readlines()	
	for i in range(1,len(csv_reader)):
		data = csv_reader[i].split(',')
		listCountInout[data[0]] = int(data[6])
		listCountElectric[data[0]] = int(data[7])
		listSerialNo[data[0]] = data[14].strip()
		listCountContruction[data[0]] = int(data[5])
		listMaker[data[0]] = int(data[4])
	errors_data = {}
	# charErr = []
	# imagePaths = ['LK_image_from_pdf/LK-11S6-02_page0.jpeg', 'LK_image_from_pdf/LK-22VC-02_page0.jpeg', 'LK_image_from_pdf/LK-32VHU-02_page0.jpeg', 'LK_image_from_pdf/LK-F32S6T EUR_page0.jpeg', 'LK_image_from_pdf/LK-F32TCT EUR_page0.jpeg', 'LK_image_from_pdf/LK-F47S6-04F (2)_page0.jpeg', 'LK_image_from_pdf/LK-F47S6-04F_page0.jpeg', 'LK_image_from_pdf/MFG No.032200723 LK-11VC-02_page0.jpeg', 'LK_image_from_pdf/MFG No.032209239 LK-21VSU-02_page0.jpeg', 'LK_image_from_pdf/MFG No.032209259 LK-F32S6T EUR_page0.jpeg', 'LK_image_from_pdf/MFG No.032211488 LK-F45TCT EUR_page0.jpeg', 'LK_image_from_pdf/MFG No.032212693 LK-21VHU-02_page0.jpeg']
	for imagePath in imagePaths:
		print("path " ,  imagePath)
		# imagePath = "LK_image_from_pdf/LK-F32S6T EUR_page0.jpeg"
		basename = os.path.basename(imagePath)
		image = cv2.imread(imagePath)
		index_in_out , index_electric, index_contruction, index_maker   = model.checkSelection(image)
		print(index_in_out , index_electric, index_contruction, index_maker)
		if listCountInout[basename] != index_in_out:
			print("basename errors " , basename)
			count_index_in_out +=1
			errors_data[basename] = 1
		if listCountElectric[basename] != index_electric:
			count_index_electric +=1
			errors_data[basename] = 2
		if listCountContruction[basename] != index_contruction:
			count_index_contruction +=1
			errors_data[basename] = 3
		if listMaker[basename] != index_maker:
			count_maker +=1
			errors_data[basename] = 4
		img_serial , serial_number= model.getSerialForm(image)
		# with open("maker_detection.csv", 'a') as f:
		# 	f.write(f'{basename},{index_maker}\n')
		print("serial_number " , serial_number , listSerialNo[basename])
		if listSerialNo[basename] != serial_number:
			count_SerialNo += 1
			errors_data[basename] = 5
			# charErr.append(basename)
			# charErr.append(listSerialNo[basename])
			# charErr.append(serial_number)
		# print("info " , index_in_out, index_electric, serial_number, listSerialNo[basename])
		print("=================================\n")
		
		# image = resize_image_min(image,input_size=1280 )
		# cv2.rectangle(image,(box_info[0], box_info[1]),(box_info[2], box_info[3]),(255,0,0),1)
		# cv2.rectangle(image,(box_info3[0], box_info3[1]),(box_info3[2], box_info3[3]),(255,0,0),2)
		path_out =os.path.join(folder_save , f'{serial_number}_{index_in_out}_{index_electric}_{index_contruction}_{index_maker}_{basename}')
		cv2.imwrite(path_out, image)
		# exit()
	print("errors " ,count_index_in_out , count_index_electric, count_index_contruction, count_maker , count_SerialNo, errors_data)
	# print(charErr)