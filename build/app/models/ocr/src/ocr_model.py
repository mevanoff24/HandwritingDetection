import tensorflow as tf 
import os
from os.path import join
import json
import random
import itertools
import re
import datetime
import numpy as np
from scipy import ndimage
import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Reshape, Lambda
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing import image
import tensorflow.keras.callbacks
import cv2
from PIL import Image
import numpy as np
import pandas as pd

import boto3

from config import *
from generator import TextImageGenerator, labels_to_text, text_to_labels



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(word_level_train, word_level_test, img_w, letters=None, sample_size=None, save=None):
    
    img_h = 64

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
        
    batch_size = 32
    downsample_factor = pool_size ** 2
    if sample_size:
        word_level_train = word_level_train.sample(sample_size)
    train_tiger = TextImageGenerator(word_level_train, data_path, image_width, image_height, batch_size, 
                               downsample_factor, max_text_len, pre_pad=False, letters=letters)
    train_tiger.build_data()
    
    if sample_size:
        word_level_test = word_level_test.sample(sample_size)
    test_tiger = TextImageGenerator(word_level_test, data_path, image_width, image_height, batch_size, 
                               downsample_factor, max_text_len, pre_pad=False, letters=letters) # update letters 
    test_tiger.build_data()
    
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
#     print(inner.shape)
    
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
#     print(inner.shape)
    
    gru1 = Bidirectional(GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1'))(inner)
    gru2 = Bidirectional(GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1'))(gru1)
    inner = Dense(train_tiger.get_output_size(), kernel_initializer='he_normal', name='dense2')(gru2)
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()
    
    labels = Input(name='the_labels', shape=[train_tiger.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
#     optimizer = Adam(lr=0.05)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
#     print(model.summary())
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])
    print(test_tiger.N, train_tiger.N)
    model.fit_generator(generator=train_tiger.next_batch(), 
                        steps_per_epoch=train_tiger.N,
                        validation_data=test_tiger.next_batch(), 
                        validation_steps=test_tiger.N,
                        epochs=epochs)
    if save_path:
    	print('Saving model to {}'.format(save_path))
    	model.save(save_path)


    return model





if __name__ == '__main__':
	word_level_train = pd.read_csv('../../../../../data/preprocessed/word_level_train.csv')
	word_level_test = pd.read_csv('../../../../../data/preprocessed/word_level_test.csv')
	data_path = '../../../../../data/raw/word_level'
	# sort by token length
	word_level_train['token_len'] = word_level_train.token.apply(len)
	word_level_train.sort_values('token_len', inplace=True)
	word_level_train = word_level_train[word_level_train['token_len'] == 5]

	word_level_test['token_len'] = word_level_test.token.apply(len)
	word_level_test.sort_values('token_len', inplace=True)
	word_level_test = word_level_test[word_level_test['token_len'] == 5]

	sess = tf.Session()
	K.set_session(sess)

	model = get_model(word_level_train, word_level_test, 128, letters=letters, sample_size=1000)





