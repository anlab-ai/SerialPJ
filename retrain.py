# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")
from keras.optimizers import SGD
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
import argparse
import random
import cv2
import os
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM , BatchNormalization , Conv2D
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tqdm import tqdm
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
set_session(sess)
batch_size = 256
num_classes = 36
epochs = 200
img_rows, img_cols = 28, 28
input_size = (img_cols,img_rows,1)
# input image dimensions

folder_data_train = "/media/anlab/ssd_samsung_256/dungtd/EMNIST/source/train_aug_crop0617/train_aug_crop0617"
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
	
	m = cv2.mean(image, th)
	# print("mean " , m)
	output = np.zeros((h,w,3), image.dtype)
	output[:, :] = (1.1*m[0], 1.1*m[1] ,1.1* m[2])
	
	kernel = np.ones((3, 3), np.uint8)
	mask = cv2.erode(mask, kernel) 
	mask = cv2.GaussianBlur(mask, (5,5), 3)
	blended = alphaBlend( output, image, mask)
	# cv2.imwrite("blended.jpg" , th)
   
	return blended



def create_mask(image, est=0.95):
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


def pre_processing(image):
	# image = resize_image(image)
	# image, mask = create_mask(image)
	# image = get_background(image, mask)
	
	gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (img_rows, img_cols))
	# gray = cv2.bitwise_not(gray)
	return gray
# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

loaded_model = Sequential()
loaded_model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (input_size[0], input_size[1], input_size[2])))
loaded_model.add(BatchNormalization())
loaded_model.add(Conv2D(32, kernel_size = 3, activation='relu'))
loaded_model.add(BatchNormalization())
loaded_model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
loaded_model.add(BatchNormalization())
loaded_model.add(Dropout(0.4))

loaded_model.add(Conv2D(64, kernel_size = 3, activation='relu'))
loaded_model.add(BatchNormalization())
loaded_model.add(Conv2D(64, kernel_size = 3, activation='relu'))
loaded_model.add(BatchNormalization())
loaded_model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
loaded_model.add(BatchNormalization())
loaded_model.add(Dropout(0.4))

loaded_model.add(Conv2D(128, kernel_size = 4, activation='relu'))
loaded_model.add(BatchNormalization())
loaded_model.add(Flatten())
loaded_model.add(Dropout(0.4))
loaded_model.add(Dense(num_classes, activation='softmax'))


# # initialize the model
# print("[INFO] compiling model...")
# json_file = open('models/model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("models/model_weights.h5")
# print("Loaded model from disk")
loaded_model.summary()
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(folder_data_train)))
random.seed(64)
random.shuffle(imagePaths)
# loop over the input images
for i, imagePath in tqdm(enumerate(imagePaths)):
	# print("Path : ", imagePath)
	# load the image, pre-process it, and store it in the data list
	image_color = cv2.imread(imagePath)
	image = pre_processing(image_color)
	# print("image shape " , image.shape)
	# cv2.imshow('image',image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# image = cv2.resize(image, (img_rows, img_cols))
	# image = cv2.equalizeHist(image)
	# image = common.preprocess2(image)
	label = imagePath.split(os.path.sep)[-2]
	label = int(label)
	if label < 0 or label > 35:
		print("label" , label)
		exit()
	labels.append(label)


	image = img_to_array(image)
	data.append(image)
	# extract the class label from the image path and update the
	# labels list
# exit()
# print("label : ", label)
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
												  labels, test_size=0.20, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=num_classes)
testY = to_categorical(testY, num_classes=num_classes)



"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
				 activation='relu',
				 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
"""


print("#layers : ", len(loaded_model.layers))
for layer in loaded_model.layers:
	print(layer)

# for layer in loaded_model.layers[:5]:
# 	layer.trainable = False

# loaded_model.compile(loss=keras.losses.categorical_crossentropy,
# 					 optimizer=keras.optimizers.Adadelta(),
# 					 metrics=['accuracy'])
loaded_model.compile(optimizer=SGD(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])
loaded_model.fit(trainX, trainY,
				 batch_size=batch_size,
				 epochs=epochs,
				 verbose=1,
				 validation_data=(testX, testY))

score = loaded_model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = loaded_model.to_json()
with open("models/model_0621.json", "w") as json_file:
   json_file.write(model_json)

loaded_model.save("models/model_0621.h5")
# serialize weights to HDF5
loaded_model.save_weights("models/model_weights_0621.h5")
print("Saved model to disk")
