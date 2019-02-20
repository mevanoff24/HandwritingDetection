from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess
# from models.OCRBeamSearch.src.SamplePreprocessor import preprocess


chars = [' ', '!', '"', '#', "'", '(', ')', '*', ',', '-', '.', '/', '&', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def new_word_level_df(word_level_df, data_path):
    """
    update path in word_level_df with new data path
    """
    word_level_df['image_path'] = word_level_df['image_path'].map(lambda x: 
                                                    data_path + x.split('word_level')[-1])
    return word_level_df

def create_samples(word_level_df):
    """create samples (image path and token)"""
    samples = []
    for path, word in zip(word_level_df.image_path, word_level_df.token):
        samples.append(Sample(word, path))
    return samples


class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath


class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts


class DataLoader:
    """
    Loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    
    Attributes: 
        dataAugmentation (boolean): Apply data augmentation to input images if true
        currIdx (int): Current row index
        batchSize (int): Size of training dataset per iteration
        imgSize (tuple): Height and width of desired image
        trainSamples (list): Training datsaet [(Sample(word, path)), ...]
        validationSamples (list): Training datsaet [(Sample(word, path)), ...]
        trainWords (list): Training text
        validationSamples (list): Validation text
        numTrainSamplesPerEpoch (int): Number of training samples per epoch
        charList (list): List of individual characters
    """ 

    def __init__(self, data_path, word_level_train, word_level_test, batchSize, imgSize, maxTextLen):
        """
        Loader for dataset at given location, preprocess images and text according to parameters
        
        Args: 
            data_path (str): Path to dataset
            word_level_train (pandas dataframe): Training set dataframe
            word_level_test (pandas dataframe): Testing set dataframe
            maxTextLen (int): Max text length to use for model
        """

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize

        word_level_train = new_word_level_df(word_level_train, data_path)
        #Drop two bad rows
        word_level_train.drop([52114, 87331], axis=0, inplace=True)
        word_level_test = new_word_level_df(word_level_test, data_path)

        self.trainSamples = create_samples(word_level_train)
        self.validationSamples = create_samples(word_level_test)
        
        # put words into lists
        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        # number of randomly chosen samples per epoch for training 
        self.numTrainSamplesPerEpoch = 25000 
        
        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))


    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input 
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i-1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text


    def trainSet(self):
        "switch to randomly chosen subset of training set"
        self.dataAugmentation = True
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

    
    def validationSet(self):
        "switch to validation set"
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples


    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


    def hasNext(self):
        "iterator"
        return self.currIdx + self.batchSize <= len(self.samples)


    def getNext(self):
        "iterator"
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)

