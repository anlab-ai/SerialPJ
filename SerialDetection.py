# set the matplotlib backend so figures can be saved in the background
from itertools import count
from posixpath import split
from tkinter.tix import Tree
import matplotlib
# matplotlib.use("Agg")
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
import imutils
from table_detection import getTable, drawTable, drawTable2

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
		json_file = open('models/model_0707_crop.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		self.loaded_model.load_weights("models/model_weights_0707_crop.h5")
		print("Loaded model from disk")
		# self.loaded_model.summary()
		self.img_debug = None
		with open('model_name.txt','r') as f:
			labels = [line.strip('\n') for line in f.readlines()]
		self.labels = labels[0].split(",")
		self.image_size = 1280
		weights_path = 'models/model_24062022.hdf5'
		self.net = get_model( weights_path = weights_path)
		self.net.summary()
		images = []
		for i in range(2):
			im = cv2.imread(f"template_data/side_{i+1}.png", 0)
			images.append(im)
		for i in range(2):
			im = cv2.imread(f"template_data/electric_type_{i+1}.png", 0)
			images.append(im)
		for i in range(3):
			im = cv2.imread(f"template_data/contruction_{i+1}_0.png", 0)
			images.append(im)

		for i in range(4):
			im = cv2.imread(f"template_data/maker_{i+1}.png", 0)
			images.append(im)
		# im1 = cv2.imread("template_data/1.png", 0)
		# im2 = cv2.imread("template_data/2.png", 0)
		# im3 = cv2.imread("template_data/3.png", 0)
		# im4 = cv2.imread("template_data/4.png", 0)
		self.feature_template = []
		
		# im1 = cv2.GaussianBlur(im1, (5,5), 0)
		# im2 = cv2.GaussianBlur(im2, (5,5), 0)
		# im3 = cv2.GaussianBlur(im3, (5,5), 0)
		# im4 = cv2.GaussianBlur(im4, (5,5), 0)
		# claheFilter = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
		# im1 = claheFilter.apply(im1)
		# im2 = claheFilter.apply(im2)
		# im3 = claheFilter.apply(im3)
		# im4 = claheFilter.apply(im4)
		features= self.getFeature_update(images)
		for i in range(len(images)):
			fea = features[i]
			fea = np.sum(fea, axis=0)
			
			# fea = fea.flatten()
			fea = fea / np.sqrt(np.sum(fea**2))
			self.feature_template.append(fea)
			# exit()
		# im = cv2.GaussianBlur(im, (3,3), 0)
		# fea = self.getFeature( im)
		# self.feature_template.append(fea)
		# im = cv2.imread("template_data/2.png", 0)
		# # im = cv2.GaussianBlur(im, (3,3), 0)
		# fea = self.getFeature(im)
		# self.feature_template.append(fea)

		# im = cv2.imread("template_data/3.png", 0)
		# # im = cv2.GaussianBlur(im, (3,3), 0)
		# fea = self.getFeature(im)
		# self.feature_template.append(fea)
		# im = cv2.imread("template_data/4.png", 0)
		# # im = cv2.GaussianBlur(im, (3,3), 0)
		# fea = self.getFeature(im)
		# self.feature_template.append(fea)
		
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
	def groupByColor(self,img_serial,listChar,color,est):
		listR = []
		listG = []
		listB = []
		listb = []
		h, w = img_serial.shape[:2]
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			im_char = img_serial[ymin:ymax, xmin:xmax]
			# plt.imshow(im_char)
			# plt.show()
			sumBlack = 0
			sumGreen = 0
			sumBlue = 0
			sumRed = 0
			im_char = Contours.convertColorToWhiteColor(im_char,color)
			w_char,h_char = im_char.shape[1],im_char.shape[0]
			for loop1 in range(h_char):
				for loop2 in range(w_char):
					r,g,b = im_char[loop1,loop2]
					if (g/(r+1) > 1.1 and g > 80 and g>b):
						sumGreen += 1
					elif (r/(g+1) > 1.1 and r > 150 and r>b):
						sumRed += 1
					elif (b/(r+1) > 1.1 and b > 130 and b>g):
						sumBlue += 1
					elif r <220 and g<220 and b <220:
						sumBlack += 1
			mainColor = np.argmax([sumRed,sumGreen,sumBlue,sumBlack])
			# print(sumRed,sumGreen,sumBlue,sumBlack)
			if mainColor == 0:
				listR.append([(xmin,ymin),(xmax,ymax)])
			elif mainColor == 1:
				listG.append([(xmin,ymin),(xmax,ymax)])
			elif mainColor == 2:
				listB.append([(xmin,ymin),(xmax,ymax)])
			else:
				listb.append([(xmin,ymin),(xmax,ymax)])
		# print(listR,listG,listB,listb)
		posMax = np.argmax([len(listR),len(listG),len(listB),len(listb)])
		if posMax == 0:
			listCharFilterColor = listR
		elif mainColor == 1:
			listCharFilterColor = listG
		elif mainColor == 2:
			listCharFilterColor = listB
		else:
			listCharFilterColor = listb
		return listCharFilterColor
	
	#Get input is image and return	MotorLotNo
	def getMotorLotNoForm(self, image):
		img = resize_image_min(image,input_size=self.image_size )
		scale = self.image_size/image.shape[1]
		box_crop = [int(scale*457),int(scale*806),int(scale*1099),int(scale*900)]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		# print("image shape", img.shape)
		imgBin,listChar,listCut = Contours.splitCharFromForm(img_serial)
		# print("listChar" , len(listChar) , listChar)
		serial_number = "" 
		est = 2
		h, w = img_serial.shape[:2]
		img_serialcp = img_serial.copy()
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			
			img_char = img_serial[ymin:ymax, xmin:xmax].copy()
			im_char_bin = imgBin[ymin:ymax, xmin:xmax]
			if i not in listCut:
				img_serial = self.removeNoise(im_char_bin,img_serial,xmin,ymin)
			# plt.imshow(img_char)
			# plt.show()
			is_digit = False
			if i != 2 and i != 3:
				is_digit = True
			idx_cls, bestclass, bestconf = self.predict(img_char , is_digit )
			serial_number += bestclass
			# cv2.rectangle(img_serialcp,(xmin, ymin),(xmax, ymax),(255,0,0),1)
			# cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/'+basename,img_serialcp)
		return img_serialcp , serial_number

	#Get input is image and return power value
	def getPowerValue(self, image):
		img = resize_image_min(image,input_size=self.image_size )
		scale = self.image_size/image.shape[1]
		box_crop = [int(scale*1730),int(scale*1215),int(scale*1950),int(scale*1290)]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		imgBin,listChar,listCut = Contours.splitCharFromForm(img_serial)
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
		trueLabels = ['440','400','220','200','380']
		img = resize_image_min(image,input_size=self.image_size )
		scale = self.image_size/image.shape[1]
		box_crop = [int(scale*1875),int(scale*1367),int(scale*2389),int(scale*1445)]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		imgcp = img_serial.copy()
		# plt.imshow(img_serial)
		# plt.show()
		imgBin,listChar,listCut = Contours.splitCharFromForm(img_serial,Color = [False, True])
		serial_number = "" 
		est = 2
		h, w = img_serial.shape[:2]
		listChar = self.groupByColor(img_serial,listChar,[False,True],est)
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			im_char = img_serial[ymin:ymax, xmin:xmax]
			m, dev = cv2.meanStdDev(im_char)
			charGray = cv2.cvtColor(im_char, cv2.COLOR_BGR2GRAY)
			ret, thresh = cv2.threshold(charGray, m[0][0] - 0.5*dev[0][0], 255, cv2.THRESH_BINARY_INV)
			# print(np.count_nonzero(thresh)/thresh.shape[0]*thresh.shape[1])
			# print('NonZero',np.count_nonzero(thresh))
			# print('area',0.1*thresh.shape[0]*thresh.shape[1])
			if np.count_nonzero(thresh) < 0.12*thresh.shape[0]*thresh.shape[1]:
				continue
			is_digit = True
			idx_cls, bestclass, bestconf = self.predict(im_char , is_digit)
			serial_number += bestclass
			# cv2.rectangle(imgcp,(xmin, ymin),(xmax, ymax),(255,0,0),1)
			# cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/'+basename,img_serial)
		# print('seraial',serial_number)
		for label in trueLabels:
			serial_numberFlag = serial_number
			while len(serial_numberFlag) != 1:
				if label.find(serial_numberFlag,0,len(serial_numberFlag)) != -1:
					return imgcp,label
				elif len(serial_numberFlag) >2 and serial_numberFlag.find(label) != -1:
					return imgcp,label		
				else:
					serial_numberFlag = serial_numberFlag[:-1]			
		return imgcp , serial_number

	#Get input is image and return Hz value
	def getHzValue(self, image):	
		trueLabels = ['50','60']
		img = resize_image_min(image,input_size=self.image_size )
		scale = self.image_size/image.shape[1]
		box_crop = [int(scale*1875),int(scale*1450),int(scale*2389),int(scale*1523)]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		# plt.imshow(img_serial)
		# plt.show()
		imgcp = img_serial.copy()
		imgBin,listChar,listCut = Contours.splitCharFromForm(img_serial)
		est = 2
		serial_number = "" 
		h, w = img_serial.shape[:2]
		listChar = self.groupByColor(img_serial,listChar,[False,True],est)
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1]-est)
			xmax = min(w , box[1][0]+est)
			ymax = min(h , box[1][1]+est)
			im_char = img_serial[ymin:ymax, xmin:xmax]
			is_digit = True
			idx_cls, bestclass, bestconf = self.predict(im_char , is_digit )
			serial_number += bestclass
			# cv2.rectangle(imgcp,(xmin, ymin),(xmax, ymax),(255,0,0),1)
			# cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/'+basename,img_serial)
		for label in trueLabels:
			serial_numberFlag = serial_number
			while len(serial_numberFlag) != 1:
				if label.find(serial_numberFlag,0,len(serial_numberFlag)) != -1:
					return imgcp,label
				elif len(serial_numberFlag) >1 and serial_numberFlag.find(label) != -1:
					return imgcp,label		
				else:
					serial_numberFlag = serial_numberFlag[:-1]		
		return imgcp , serial_number

	#Get input is image and return min value
	def getMinValue(self, image):	
		img = resize_image_min(image,input_size=self.image_size )
		scale = self.image_size/image.shape[1]
		box_crop = [int(scale*1885),int(scale*1538),int(scale*2389),int(scale*1625)]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		imgcp = img_serial.copy()
		imgBin,listChar, listCut = Contours.splitCharFromForm(img_serial,params=[[9,0.25]],num=4)
		serial_number = "" 
		est = 2
		h, w = img_serial.shape[:2]
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			
			img_char = img_serial[ymin:ymax, xmin:xmax].copy()
			im_char_bin = imgBin[ymin:ymax, xmin:xmax]
			if i not in listCut:
				img_serial = self.removeNoise(im_char_bin,img_serial,xmin,ymin)
			# plt.imshow(img_serial)
			# plt.show()
			is_digit = True
			img_char = cv2.GaussianBlur(img_char, (3,3), 3)
			idx_cls, bestclass, bestconf = self.predict(img_char , is_digit)
			serial_number += bestclass
			# cv2.rectangle(imgcp,(xmin, ymin),(xmax, ymax),(255,0,0),1)
			# cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/'+basename,img_serial)
		return imgcp , serial_number
	def get_backgroundColor(self,image):
		gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
		m = cv2.mean(gray)
		_, th = cv2.threshold(gray,m[0],255,cv2.THRESH_BINARY)
		m = cv2.mean(image, th)
		return 0.97*m[0], 0.97*m[1] ,0.97* m[2]
	def removeNoise(self, im_char_bin,img_serial,x_min,y_min):
		contours,hierachy=cv2.findContours(im_char_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		# contours = contours[0] if imutils.is_cv2() else contours[1]  
		# cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
		r,g,b = self.get_backgroundColor(img_serial)
		for contour in contours:
			for value in contour:
				value[0][0]+=x_min
				value[0][1]+=y_min
		cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
		# plt.imshow(img_serial)
		# plt.show()
		# if len(contours)>1:
		cv2.drawContours(img_serial, [cntsSorted[0]], -1, (r, g, b), -1)
		# plt.imshow(img_serial)
		# plt.show()
		return img_serial
	#Get input is image and return serial number
	def getSerialForm(self, image):
		img = resize_image_min(image,input_size=self.image_size )
		box_crop = [968,830, 1233, 887]
		img_serial = img[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]
		img_serialcp = img_serial.copy()
		# print("image shape", img.shape)
		imgBin,listChar,listCut = Contours.splitCharFromForm(img_serial)
		# print("listChar" , len(listChar) , listChar)
		serial_number = "" 
		est = 2
		h, w = img_serial.shape[:2]
		for i, box in enumerate(listChar):
			xmin = max(0 , box[0][0]-est)
			ymin = max(0 , box[0][1] -est)
			xmax = min(w , box[1][0] + est)
			ymax = min(h , box[1][1] + est)
			
			img_char = img_serial[ymin:ymax, xmin:xmax].copy()
			im_char_bin = imgBin[ymin:ymax, xmin:xmax]
			if i not in listCut:
				img_serial = self.removeNoise(im_char_bin,img_serial,xmin,ymin)
			# plt.imshow(img_char)
			# plt.show()
			
			is_digit = False
			if i  > 0  :
				is_digit = True
			idx_cls, bestclass, bestconf = self.predict(img_char , is_digit )
			serial_number += bestclass
		# for i, box in enumerate(listChar):
		# 	xmin = box[0][0]
		# 	ymin = box[0][1]
		# 	xmax = box[1][0]
		# 	ymax = box[1][1]	
			cv2.rectangle(img_serialcp,(xmin, ymin),(xmax, ymax),(255,0,0),1)
		
		return img_serialcp , serial_number

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

	def getSelection(self, image, threshold_box_min = 0.5, threshold_box_max = 0.8, thresh_count_line = 0.35):
		h, w = image.shape[:2]
		is_Selection = False
		image_cv = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		img_clean = Contours.convertColorToWhiteColor(image_cv)
		gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
		# Inverse 
		m, dev = cv2.meanStdDev(gray)
		ret, thresh = cv2.threshold(gray, m[0][0] - 0.5*dev[0][0], 255, cv2.THRESH_BINARY_INV)
		thresh = Contours.delLine(thresh)
		h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	
		thresh = cv2.dilate(thresh,h_structure,1)
		contours,hierachy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
		# plt.imshow(image)
		# plt.show()
		ratio_line = 1
		ratio_box = 0
		if len(contours) > 0 :
			box_max = cv2.boundingRect(contours[0])
			area_max = cv2.contourArea(contours[0])
			for cnt in contours:
				area = cv2.contourArea(cnt)
				if area_max < area:
					area_max = area
					box_max = cv2.boundingRect(cnt)
			img_line = thresh[box_max[1] :box_max[1] + box_max[3] ,box_max[0]  :box_max[0] + box_max[2]  ]
			count_line = cv2.countNonZero(img_line)
			count_line2 = cv2.countNonZero(thresh)
			area_box_line = box_max[2]*box_max[3]
			ratio_line = count_line/area_box_line
			ratio_box = area_box_line/(w*h)
			if count_line2 / (count_line +1 ) < 1.2:
				
				if ratio_box < threshold_box_min or ratio_line > thresh_count_line:
					is_Selection = True
					if ratio_box > threshold_box_max and ratio_line < 1.3*thresh_count_line:
						is_Selection = False
			else:
				is_Selection = True
		# 	print( "here ",  is_Selection , ratio_box ,ratio_line )
		# cv2.imwrite("image_clean.jpg" , thresh )
		# exit()
		return is_Selection , contours , ratio_line, ratio_box

	def getCheckTable(self , image):
		ret = False
		img = resize_image_min(image,input_size=self.image_size )
		scale = image.shape[0]/img.shape[0]
		box = [733, 381 ,895, 1086 ]
		img = img[box[1]:box[3], box[0]:box[2]]
		table = getTable(img)
		# print("table " , len(table))
		
		est = 3
		status_table = []
		max_check = min(20,len(table)) 
		table = table[0:max_check]
		ctn = None
		ratio_line_ct = 1
		ratio_box_ct = 0
		index = -1
		for i, tb in enumerate(table):
			# if i != 1 :
			# 	continue
			tb = tb[0]
			img_select = img[tb[1] +est:tb[1] + tb[3] -2*est ,tb[0] +est :tb[0] + tb[2] -2 *est ]
			is_Selection, ct , ratio_line, ratio_box = self.getSelection(img_select)
			
			if i >= 1 and is_Selection and len(ct) == 1:
				if len(ctn) == 1:
					(center1, (w1, h1), al1) = cv2.minAreaRect(ct[0])
					(center2, (w2, h2), al2) = cv2.minAreaRect(ctn[0])
					b1 = cv2.boundingRect(ct[0])
					b2 = cv2.boundingRect(ctn[0])
					distance_line = abs(b2[0] - b1[2] - b1[0])
					# print("angle " , i, al1 , al2 ,distance_line ,distance_line )
					if al1 > 25 and al1 <65 and al2 > 25 and al2 <65 and distance_line < 12:
						is_Selection = False
						status_table[len(status_table) -1 ] = False
			
			ctn = ct
			ratio_line_ct = ratio_line
			ratio_box_ct = ratio_box
			status_table.append(is_Selection)
		output = drawTable(img, table , status_table)
		# cv2.imwrite("t1.jpg" ,output )
		if max_check == 20:
			ret = True
		else:
			status_table = []
		return ret , status_table , output

	def checkSign(self, image):
		h, w = image.shape[:2]
		numData = 0
		image_cv = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		img_clean = Contours.convertColorToWhiteColor(image_cv)
		gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
		# Inverse 
		m, dev = cv2.meanStdDev(gray)
		ret, thresh = cv2.threshold(gray, m[0][0] - 0.5*dev[0][0], 255, cv2.THRESH_BINARY_INV)
		thresh = Contours.delLine(thresh)
		h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	
		thresh = cv2.dilate(thresh,h_structure,1)
		contours,hierachy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		# cv2.imwrite("t1.jpg" ,thresh )
		# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
		# plt.imshow(image)
		# plt.show()
		ratio_line = 1
		ratio_box = 0
		if len(contours) > 0 :
			for cnt in contours:
				box_max = cv2.boundingRect(cnt)
				area = box_max[2]*box_max[3]
				if area < 30 :
					continue
				numData +=1
		return numData
 
	def getSignName(self , image, length_data=2 , leng_sign = 0):
		
		results = [False, False]
		ret = False
		img = resize_image_min(image,input_size=self.image_size )
		scale = image.shape[0]/img.shape[0]
		box = [451, 1262 ,1095, 1343 ]
		img = img[box[1]:box[3], box[0]:box[2]]
		
		table = getTable(img, threshold_size=0.45)
		output = drawTable2(img, table )
		# cv2.imwrite("t1.jpg" ,output )
		isDetectTable = False
		box1 = None
		box2 = None
		if len(table) >= 2 :
			isDetectTable = True
			box1 = table[0][0]
			box2 = table[1][0]
		elif len(table) == 1:
			if len(table[0]) >= 2:
				isDetectTable = True
				box1 = table[0][0]
				box2 = table[0][1]
		if isDetectTable:
			est = 3
			img_select = img[box1[1] +est:box1[1] + box1[3] -2*est ,box1[0] + box1[2] // 3+est :box1[0] + box1[2] -2 *est ]
			numData = self.checkSign(img_select)
			
			if numData > length_data:
				results[0] = True
			img_select = img[box2[1] +est:box2[1] + box2[3] -2*est ,box2[0] + box2[2] // 2 :box2[0] + box2[2] -2 *est ]
			numData2 = self.checkSign(img_select)
			if numData2 > leng_sign:
				results[1] = True
			print("numData " , numData , numData2)
		print("table " , table ,results )
		return results

if __name__ == '__main__':

	model = SerialDetection()
	imagePaths = sorted(list(paths.list_images("jpg_files_LK-20220815T025511Z-001")))
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
	list_status = {}
	with open('expected_result_70_images_update.csv') as csv_file:
			csv_reader = csv_file.readlines()
	with open('status_results.csv') as csv_file:
		status_reader = csv_file.readlines()
	for l in status_reader:
		data = l.split(',')
		li = []
		
		for i in range(1,len(data)):
			
			if data[i].strip() == "TRUE" :
				li.append(True)
			else:
				li.append(False)
		
		list_status[data[0]] = li
	# 	key_mfg = data[0].split(" ")[0]
	# 	print("key_mfg " , key_mfg)
	# 	if key_mfg == "MFG":
	# 		l_out = l.replace("_page0.jpeg",".jpg")
	# 		with open('update_stt.csv', 'a') as f:
	# 			f.write(f'{l_out}')

	# exit()
	# print("imagePaths " , imagePaths)
	for i in range(1,len(csv_reader)):
		data = csv_reader[i].split(',')
		listCountInout[data[0]] = int(data[6])
		listCountElectric[data[0]] = int(data[7])
		listSerialNo[data[0]] = data[14].strip()
		listCountContruction[data[0]] = int(data[5])
		listMaker[data[0]] = int(data[4])
	errors_data = {}
	errors_count_stt = []
	errors_count_stt2 = []
	# charErr = []
	# imagePaths = ['LK_image_from_pdf/LK-11S6-02_page0.jpeg', 'LK_image_from_pdf/LK-22VC-02_page0.jpeg', 'LK_image_from_pdf/LK-32VHU-02_page0.jpeg', 'LK_image_from_pdf/LK-F32S6T EUR_page0.jpeg', 'LK_image_from_pdf/LK-F32TCT EUR_page0.jpeg', 'LK_image_from_pdf/LK-F47S6-04F (2)_page0.jpeg', 'LK_image_from_pdf/LK-F47S6-04F_page0.jpeg', 'LK_image_from_pdf/MFG No.032200723 LK-11VC-02_page0.jpeg', 'LK_image_from_pdf/MFG No.032209239 LK-21VSU-02_page0.jpeg', 'LK_image_from_pdf/MFG No.032209259 LK-F32S6T EUR_page0.jpeg', 'LK_image_from_pdf/MFG No.032211488 LK-F45TCT EUR_page0.jpeg', 'LK_image_from_pdf/MFG No.032212693 LK-21VHU-02_page0.jpeg']
	for imagePath in imagePaths:
		# imagePath = "LK_image_from_pdf/MFG No.032209259 LK-F32S6T EUR_page0.jpeg"
		# imagePath = "jpg_files_LK-20220815T025511Z-001/jpg_files_LK/MFG No.042214837 LK-F45S6T-E02.jpg"
		print("path " ,  imagePath)
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
		# print("info " , index_in_out, index_electric, serial_number, listSerialNo[basename]
		result_sign = model.getSignName(image)
		print("result_sign " , result_sign)
		if not result_sign[0] or  not  result_sign[1]:
			errors_count_stt.append(basename)
		ret , status_table , im_table = model.getCheckTable(image)
		str = {basename}
		# result_stt = []
		result_stt =  list_status[basename.strip()]
		print("result_stt ", result_stt, status_table)
		if ret:
			for j , s in enumerate(status_table):
				if s != result_stt[j]:
					ret = False
					print("index " , j , s , result_stt[j])
				str = f'{str},{s}'
		if not ret :
			errors_count_stt2.append(basename)
		print("status_table " , ret)
		with open('status.csv', 'a') as f:
			
			f.write(f'{str}\n')
		print("=================================\n")
		path_out =os.path.join(folder_save , f'{serial_number}_{index_in_out}_{index_electric}_{index_contruction}_{index_maker}_{basename}')
		cv2.imwrite(path_out, im_table)
		# exit()

	print("errors " ,errors_count_stt , errors_count_stt2)
	# print(charErr)
