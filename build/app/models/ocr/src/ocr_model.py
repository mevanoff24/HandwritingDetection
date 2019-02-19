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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
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


def get_model(word_level_train, word_level_test, img_w, max_text_len, train_samples=None, test_samples=None, letters=None, 
												sample_size=None, save_path=None, use_s3=False):
	

	if K.image_data_format() == 'channels_first':
		input_shape = (1, img_w, img_h)
	else:
		input_shape = (img_w, img_h, 1)
		
	if sample_size:
		word_level_train = word_level_train.sample(sample_size)
	train_tiger = TextImageGenerator(word_level_train, data_path, img_w, img_h, batch_size, 
			downsample_factor, max_text_len, pre_pad=False, letters=letters, samples=train_samples, use_s3=use_s3)
	train_tiger.build_data()
	
	if sample_size:
		word_level_test = word_level_test.sample(sample_size)
	test_tiger = TextImageGenerator(word_level_test, data_path, img_w, img_h, batch_size, 
			downsample_factor, max_text_len, pre_pad=False, letters=letters, samples=test_samples, use_s3=use_s3) # update letters 
	test_tiger.build_data()
	
	act = activation
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
	
	inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
	
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
	
	sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

	model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
	# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
	
	# captures output of softmax so we can decode the output during visualization
	test_func = K.function([input_data], [y_pred])
	print('Number of training and testing  images: {} -- {}'.format(train_tiger.N, test_tiger.N))
    
	filepath = "../models/weights-improvement2-10-{epoch:02d}-{val_loss:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001)
	callbacks_list = [checkpoint, reduce_lr]


	model.fit_generator(generator=train_tiger.next_batch(), 
						steps_per_epoch=train_tiger.N,
						validation_data=test_tiger.next_batch(), 
						validation_steps=test_tiger.N,
						epochs=epochs,
						callbacks=callbacks_list)
	if save_path:
		print('Saving model to {}'.format(save_path))
		model.save(save_path)


	return model




def create_s3_samples(bucket, S3_IMAGE_PATH, word_level_train, word_level_test):
	samples_train = []
	samples_test = []
	print('Getting S3 Image files')
	files = list(bucket.objects.filter(Prefix=S3_IMAGE_PATH))
	cant_find = 0
	for file in files:
		if '.png' in file.key:
			try:
				token = word_level_train.loc[word_level_train.image_path == file.key, 'token'].tolist()[0]
				samples_train.append([file, token])
			except:
				cant_find += 1
				pass
			
			try:
				token = word_level_test.loc[word_level_test.image_path == file.key, 'token'].tolist()[0]
				samples_test.append([file, token])
			except:
				cant_find += 1
				pass
	print(cant_find)
	print('Done! train_size: {}, test_size: {}'.format(len(samples_train), len(samples_test)))
	return samples_train, samples_test

def subset_data(word_level_df, greater_than_val, less_than_val):
    word_level_df['token_len'] = word_level_df.token.apply(len)
    word_level_df.sort_values('token_len', inplace=True)
    word_level_df = word_level_df[(word_level_df['token_len'] >= greater_than_val) & (word_level_df['token_len'] < less_than_val)]
    return word_level_df

if __name__ == '__main__':
    word_level_train = pd.read_csv('../../../../../data/preprocessed/word_level_train.csv')
    word_level_test = pd.read_csv('../../../../../data/preprocessed/word_level_test.csv')
    data_path = '../../../../../data/raw/word_level'

    if USE_S3:
        S3_WORD_LEVEL_TRAIN_PATH = 'data/word_level_train.csv'
        S3_WORD_LEVEL_TEST_PATH = 'data/word_level_test.csv'
        S3_IMAGE_PATH = 'data/word_level'
        S3_BUCKET = 'handwrittingdetection'
        data_path = 'data/word_level'
        client = boto3.resource('s3')
        bucket = client.Bucket(S3_BUCKET)
        word_level_train = pd.read_csv(os.path.join('s3n://', S3_BUCKET, S3_WORD_LEVEL_TRAIN_PATH))
        word_level_test = pd.read_csv(os.path.join('s3n://', S3_BUCKET, S3_WORD_LEVEL_TEST_PATH))
        word_level_train['image_path'] = word_level_train['image_path'].map(lambda x: data_path + x.split('word_level')[-1])
        word_level_test['image_path'] = word_level_test['image_path'].map(lambda x: data_path + x.split('word_level')[-1])
        
        word_level_train = subset_data(word_level_train, greater_than_val, less_than_val)
        word_level_test = subset_data(word_level_test, greater_than_val, less_than_val)
        word_level_test = word_level_test.sample(frac=1)

        samples_train, samples_test = create_s3_samples(bucket, S3_IMAGE_PATH, word_level_train, word_level_test)

    else:
        word_level_train = pd.read_csv('../../../../../data/preprocessed/word_level_train.csv')
        word_level_test = pd.read_csv('../../../../../data/preprocessed/word_level_test.csv')
        data_path = '../../../../../data/raw/word_level'
        samples_train = None
        samples_test = None
        # sort by token length
        word_level_train = subset_data(word_level_train, greater_than_val, less_than_val)
        word_level_test = subset_data(word_level_test, greater_than_val, less_than_val)


    sess = tf.Session()
    K.set_session(sess)

    model = get_model(word_level_train, word_level_test, img_w, max_text_len=max_text_len, train_samples=samples_train, 
                      test_samples=samples_test, letters=letters, sample_size=None, use_s3=USE_S3, 
                      save_path='../models/ocr_2_10_lr_01_size_128.h5')

    sess.close()


# 73616/73616 [==============================] - 6757s 92ms/step - loss: 2.1937 - val_loss: 2.9973
# Saving model to ../models/ocr_2_10_lr_01_size_128.h5

# USE_S3 = False
# img_w = 128 
# img_h = 64
# conv_filters = 16
# kernel_size = (3, 3)
# pool_size = 2
# time_dense_size = 32
# rnn_size = 128
# batch_size = 64
# downsample_factor = pool_size ** 2
# max_text_len = 10
# activation = 'relu'
# learning_rate = 0.01
# epochs = 1
# greater_than_val = 2
# less_than_val = 10

