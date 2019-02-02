from collections import defaultdict, Counter
import dill as pickle
import cv2
import os
import numpy as np
from scipy import ndimage
import random
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K


def labels_to_text(labels, letters):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text, letters):
    return list(map(lambda x: letters.index(x), text))

def is_valid_str(s, letters):
    for ch in s:
        if not ch in letters:
            return False
    return True

def unpickle(filename):
    """Unpickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

class TextImageGenerator:
    
    def __init__(self, word_level_df, data_path, img_width, img_height, batch_size, downsample_factor,
                 max_text_len=21, samples=None, pre_pad=True, use_s3=False, letters=None, is_training=True):
        
        self.data_path = data_path
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.max_text_len = max_text_len
        self.pre_pad = pre_pad
        self.is_training = is_training
        self.use_s3 = use_s3
        
        word_level_df['image_path'] = word_level_df['image_path'].map(lambda x: 
                                                    self.data_path + x.split('word_level')[-1])
        self.word_level_df = word_level_df

        if letters == None:
            self.letters = sorted(list(Counter(''.join(word_level_df.token.values)).keys()))
            self.letters.append(' ')
            self.letters = sorted(list(set(self.letters)))
        else:
            self.letters = letters
        self.pad_idx = self.letters.index(' ')
        
        # training data 
        if samples:
            self.samples = samples
        else: 
            self.samples = self.word_level_df[['image_path', 'token']].values.tolist()
        self.N = len(self.samples)
        self.current_index = 0
        
    def build_data(self):
        self.images = np.zeros((self.N, self.img_height, self.img_width))
        self.texts = []
        bad_records = []
        for i, (img_path, text) in enumerate(self.samples):
            try:
                # read image 
                if self.use_s3:
                    if '.png' in img_path.key:
                        img = img_path.get()['Body'].read()  
                        img = np.frombuffer(img, np.uint8)
                        img = cv2.imdecode(img, 1)
                else:
                    img = cv2.imread(img_path)
                # grayscale image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # resize image
                img = cv2.resize(img, (self.img_width, self.img_height))
                # change image type
                img = img.astype(np.float32)
                # scale image 
                img /= 255
                # width and height are backwards from typical Keras convention
                # because width is the time dimension when it gets fed into the RNN
                self.images[i, :, :] = img
                self.texts.append(text)
            except:
                print('Image not available for image', i, img_path, text)
                bad_records.append(i)
        # update stats to remove bad records with no image data 
        self.N -= len(bad_records)
        self.indexes = list(range(self.N))
        self.images = np.delete(self.images, bad_records, axis=0)

    def get_output_size(self):
        return len(self.letters) + 1
#         return len(all_letters) + 1

    def next_sample(self):
        self.current_index += 1
        if self.current_index >= self.N:
            self.current_index = 0
            random.shuffle(self.indexes)
        return self.images[self.indexes[self.current_index]], self.texts[self.indexes[self.current_index]]

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X = np.ones([self.batch_size, 1, self.img_width, self.img_height])
            else:
                X = np.ones([self.batch_size, self.img_width, self.img_height, 1])

            y = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_width // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X[i] = img
                y_numeric = text_to_labels(text, self.letters)
                if self.pre_pad: padded_y = ([self.pad_idx] * (self.max_text_len - len(y_numeric))) + y_numeric
                else: padded_y = y_numeric + ([self.pad_idx] * (self.max_text_len - len(y_numeric)))
                y[i] = padded_y
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X,
                'the_labels': y,
                'input_length': input_length,
                'label_length': label_length,
            }          
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)

        