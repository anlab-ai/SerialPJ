
from tensorflow.keras import backend as K
# from tensorflow.keras import Layer
from tensorflow.keras import models
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras import layers


from parameters import *
K.set_learning_phase(0)

# # Loss and train functions, network architecture
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def batch_activate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x
def convolution_block(x,
                      filters,
                      size,
                      strides=(1, 1),
                      padding='same',
                      activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = batch_activate(x)
    return x

def residual_block(block_input,
                   num_filters=16,
                   use_batch_activate=False):
    x = batch_activate(block_input)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = layers.Add()([x, block_input])
    if use_batch_activate:
        x = batch_activate(x)
    return x
def get_model( weights_path =None, start_neurons=32, dropout_rate=0.1):

    input_shape = (img_w, img_h, channels)     # (128, 64, 1)

    # Make Networkw
    inputs = layers.Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

    for index, i in enumerate([1, 2, 2, 4, 8]):
        if index == 0:
            inner = inputs
        inner = layers.Conv2D(start_neurons * i, (3,3),
                              activation=None, padding="same")(inner)
        inner = residual_block(inner, start_neurons * i)
        inner = residual_block(inner, start_neurons * i, True)

        if i <=2:
            inner = layers.MaxPooling2D((2,2))(inner)

        if dropout_rate:
            inner = layers.Dropout(dropout_rate)(inner)
    inner_shape = inner.get_shape()


# def get_model(training):
#     input_shape = (img_w, img_h, channels)     # (128, 64, 1)

#     # Make Networkw
#     inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

#     # Convolution layer (VGG)
#     inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  # (None, 128, 64, 64)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

#     inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

#     inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(2, 2), name='max3')(inner)  # (None, 32, 8, 256)

#     #Em freeze ở đây

#     inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 512)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

#     inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner_shape = inner.get_shape()


    #Tác giả freeze

    # CNN to RNN
    # inner = Reshape(target_shape=((-1, 512)), name='reshape')(inner)  # (None, 32, 2048)
    inner = layers.Reshape(target_shape=(int(inner_shape[1]), int(inner_shape[2] * inner_shape[3])), name='reshape')(inner)
    inner = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

    # RNN layer
    # lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 32, 512)
    # lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    # lstm1_merged = Add()([lstm_1, lstm_1b])  # (None, 32, 512)
    # lstm1_merged = BatchNormalization()(lstm1_merged)



    # lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    # lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
    # lstm2_merged = Concatenate()([lstm_2, lstm_2b])  # (None, 32, 1024)
    # lstm2_merged = BatchNormalization()(lstm2_merged)


    inner = layers.Bidirectional(
        layers.LSTM(units=256, return_sequences=True), name='bi_lstm1')(inner)
    inner = layers.Bidirectional(
        layers.LSTM(units=256, return_sequences=True), name='bi_lstm2')(inner)
    

    # transforms RNN output to character activations:
    inner = layers.Dense(num_classes, kernel_initializer='he_normal',name='dense2')(inner) #(None, 32, 63)
    y_pred = layers.Activation('softmax', name='softmax')(inner)

    labels = layers.Input(name='the_labels', shape=[max_text_len], dtype='float32') # (None ,8)
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)

    # if training:
    #     return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    # else:
    model = Model(inputs=[inputs], outputs=y_pred)
    # model.summary()
    # exit()
    model.load_weights(weights_path)
    model = Model(inputs=[inputs], outputs=model.layers[-3].output)
    # 
    return model

