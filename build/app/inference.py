# LM model imports 
import os
import time
import torch
from torch import optim

from models.context2vec.src.mscc_eval import mscc_evaluation
from models.context2vec.src.model import Context2vec
from models.context2vec.src.args import parse_args
from models.context2vec.src.dataset import WikiDataset
from models.context2vec.src.config import Config
from models.context2vec.src.utils import write_embedding, write_config, read_config, load_vocab


# OCR model imports 
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import itertools
from models.ocr.src.config import letters


# OCRBeamSearch model
from models.OCRBeamSearch.src.Model import Model, DecoderType
from models.OCRBeamSearch.src.SamplePreprocessor import preprocess
import sys
import cv2
import editdistance

from collections import defaultdict
from operator import itemgetter
from nltk.stem import PorterStemmer, WordNetLemmatizer
import Levenshtein
import re



class FilePaths:
    "filenames and paths to data"
    fnCharList = 'models/OCRBeamSearch/model/charList.txt'
    fnAccuracy = 'models/OCRBeamSearch/model/accuracy.txt'
    fnTrain = 'models/OCRBeamSearch/data/'
    fnInfer = 'models/OCRBeamSearch/data/test.png'
    fnCorpus = 'models/OCRBeamSearch/data/corpus.txt'


class Batch:
    "batch containing images and ground truth texts"
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


class Inference():
    """
    Model Inference
    
    Attributes:
        device (str): where to run the code - cpu or gpu
        img_width (int): desired width of image (only used for orig OCR)
        img_height (int): desired height of image (only used for orig OCR)
        stemmer (object): NLTK PorterStemmer class
        lemma (object): NLTK WordNetLemmatizer class      
    """
    def __init__(self, img_width=128, img_height=64, device='cpu', decoding=None):
        self.device = device
        self.img_width = img_width
        self.img_height = img_height
        self.decoding = decoding
        self.build_language_model()
        # self.build_ocr_model()
        self.build_beam_ocr_model(decoding)
    
        self.stemmer = PorterStemmer()
        self.lemma = WordNetLemmatizer()

        
    def build_language_model(self, model_dir='models/context2vec/models_103'): 
        """
        Builds Language model
        
        Args:
           model_dir (str): path to model directory
           
        Returns:
            None
        """
        # LANGUAGE MODEL
        modelfile = os.path.join(model_dir, 'model.param')
        wordsfile = os.path.join(model_dir, 'embedding.vec')
        config_file = modelfile+'.config.json'
        config_dict = read_config(config_file)
        self.lm_model = Context2vec(vocab_size=config_dict['vocab_size'],
                            counter=[1]*config_dict['vocab_size'],
                            word_embed_size=config_dict['word_embed_size'],
                            hidden_size=config_dict['hidden_size'],
                            n_layers=config_dict['n_layers'],
                            bidirectional=config_dict['bidirectional'],
                            dropout=config_dict['dropout'],
                            pad_idx=config_dict['pad_index'],
                            device=self.device,
                            inference=True).to(self.device)
        self.lm_model.load_state_dict(torch.load(modelfile, map_location=self.device))
        self.itos, self.stoi = load_vocab(wordsfile)
        self.unk_token = config_dict['unk_token']
        self.bos_token = config_dict['bos_token']
        self.eos_token = config_dict['eos_token']


    def build_beam_ocr_model(self, decoding):
        """
        Builds Beam Search OCR model
        
        Args:
           decoderType (str): Decoding Type for Beam Search
           
        Returns:
            None
        """
        # Beam Search OCR model
        if decoding == 'beamsearch':
            decoderType = DecoderType.BeamSearch
        if decoding == 'wordbeamsearch':
            decoderType = DecoderType.WordBeamSearch
        else:
            decoderType = DecoderType.BestPath
        self.beam_ocr_model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
        

        
    def build_ocr_model(self):
        """
        Builds Original OCR model
        """
        self.sess = tf.Session()
        K.set_session(self.sess)

        ocr_model_path = 'models/ocr/models/weights-improvement2-10-01-3.00.hdf5'
        self.ocr_model = load_model(ocr_model_path, custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})

        
    def preprocess_image(self, img_path, img_width, img_height):
        """
        Preprocess image for Original OCR model
        
        Args:
           img_path (str): Path to image
           img_width (int): desired width of image (only used for orig OCR)
           img_height (int): desired height of image (only used for orig OCR)
            
        Returns:
            img (numpy array): Scaled, formated, and reshaped image as numpy array
        """
        img = cv2.imread(img_path)
        # grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # resize image
        img = cv2.resize(img, (img_width, img_height))
        # change image type
        img = img.astype(np.float32)
        # scale image 
        img /= 255
        img = img.reshape((1, img_width, img_height, 1))
        return img
        
        
    def _decode_batch(self, out):
        """
        Best Path decoding for original OCR model
        
        Args:
           out (array): Predictions array 
            
        Returns:
            ret (str): String of maximum likihood word 
        """
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if c < len(letters):
                    outstr += letters[c]
            ret.append(outstr)
        return ret

        
    def _return_split_sentence(self, sentence):
        """
        Formats input sentence for language model
        
        Args:
           sentence (str): Input string 
            
        Returns:
            tokens (list): List of tokens
            target_pos (int): Index of target word 
        """
        if ' ' not in sentence:
            print('sentence should contain white space to split it into tokens')
            raise SyntaxError
        elif '[]' not in sentence:
            print('sentence should contain `[]` that notes the target')
            raise SyntaxError
        else:
            tokens = sentence.lower().strip().split()
            target_pos = tokens.index('[]')
            return tokens, target_pos

        
    def run_lm_inference_by_user_input(self, sentence, topK=100):
        """
        Processes user input sentence
        Runs user input sentence through model
        Returns topK predictions and probabilities
        
        Args:
           sentence (str): User input sentence 
           topK (int): Number of top predictions to return 
        
        Returns:
            output (list): list of tuples [(probability, token), ...]
        """
        # Dont return these to user 
        bad_list = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<unk>']
        # evaluation mode 
        self.lm_model.eval()
        # norm_weight
        self.lm_model.norm_embedding_weight(self.lm_model.criterion.W)

        tokens, target_pos = self._return_split_sentence(sentence)
        tokens[target_pos] = self.unk_token
        tokens = [self.bos_token] + tokens + [self.eos_token]
        indexed_sentence = [self.stoi[token] if token in self.stoi else self.stoi[self.unk_token] for token in tokens]
        # to torch tensor
        input_tokens = torch.tensor(indexed_sentence, dtype=torch.long, device=self.device).unsqueeze(0)
        # run through model
        topv, topi = self.lm_model.run_inference(input_tokens, target=None, target_pos=target_pos, k=topK)
        output = []  
        for value, key in zip(topv, topi):
            word = self.itos[key.item()]
            if word not in bad_list:
                output.append((value.item(), word))
        return output

    
    def run_beam_ocr_inference_by_user_image(self, img_path):
        """
        Processes user input image BS
        Runs user input image through model
        Returns top prediction and probability
        
        Args:
           img_path (str): Path to user uploaded image
        
        Returns:
            recognized (list): Predicted token
            probability (list): Probability of predictions
        """
        img = preprocess(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), Model.imgSize)
        batch = Batch(None, [img])
        (recognized, probability) = self.beam_ocr_model.inferBatch(batch, True)
        return (recognized, probability)

    
    def run_ocr_inference_by_user_image(self, img):
        """
        Returns top prediction from original OCR model
        
        Args:
           img (str): Path to user uploaded image
        
        Returns:
            pred_texts (list): Predicted token
        """
        net_inp = self.ocr_model.get_layer(name='the_input').input
        net_out = self.ocr_model.get_layer(name='softmax').output
        net_out_value = self.sess.run(net_out, feed_dict={net_inp: img})
        pred_texts = self._decode_batch(net_out_value)
        return pred_texts

    
    def create_features_improved(self, lm_preds, ocr_pred, ocr_prob):
        """
        Create features for weighing algorithm using 
        language and OCR models predictions and probabilities
        
        Args:
           lm_preds (list): list of tuples [(probability, token), ...]
           ocr_pred (list): Predicted token
           ocr_prob (list): Probability of predictions
        
        Returns:
            features (dict): features for each model prediction token
        """
        # create bins for length
        bins = {
            'small': list(range(0, 3)),
            'small-mid': list(range(2, 6)),
            'mid': list(range(4, 8)),
            'mid-large': list(range(6, 10)),
            'large': list(range(8, 12)),
            'large-big': list(range(10, 14)),
            'big': list(range(12, 100)),
        }

        bins = defaultdict(lambda: 'na', bins)

        ocr_len = len([x for x in ocr_pred[0]])
        ocr_pred_bins = [k for k, v in bins.items() if ocr_len in v]

        features = {}
        bad_list = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<unk>'] # ADD foul words
        matches = {}
        matches_non_ordered = {}

        ocr_pred_lower = ocr_pred[0].lower()

        for lm_pred in lm_preds:
            lm_prob, word = lm_pred[0], lm_pred[1].rstrip()
            word = word.lower()
            # remove pad, bos, etc...
            if word not in bad_list:
#                 try:
                features[word] = {}
                features[word]['ocr_prob'] = ocr_prob[0]
                features[word]['lm_prob'] = lm_prob
                # length 
                word_len = len(word)
                features[word]['exact_length_match'] = word_len == ocr_len

                word_bins = [k for k, v in bins.items() if word_len in v]
                features[word]['bin_length_match'] = False
                for bin_ in ocr_pred_bins:
                    if bin_ in word_bins:
                        features[word]['bin_length_match'] = True

                # levenshtein distance
                features[word]['levenshtein'] = Levenshtein.ratio(word, ocr_pred_lower)
                # editdistance (1 / dist) so less is better 
                features[word]['editdistance'] = 1 / (editdistance.eval(word, ocr_pred_lower) + 0.001) # for divide by zero error

                # exact match
                exact = word == ocr_pred_lower
                exact_stem = self.stemmer.stem(word) == self.stemmer.stem(ocr_pred_lower)
                exact_lemma = self.lemma.lemmatize(word) == self.lemma.lemmatize(ocr_pred_lower)
                exact_length = word == ocr_pred_lower
                features[word]['exact'] = exact
                features[word]['exact_stem'] = exact_stem
                features[word]['exact_lemma'] = exact_lemma
                # match first and last character 
                first_char_match = word[0] == ocr_pred_lower[0]
                last_char_match = word[-1] == ocr_pred_lower[-1]
                features[word]['first_char_match'] = first_char_match
                features[word]['last_char_match'] = last_char_match


                # number of character matches
                num_chars = 0
                for char in ocr_pred_lower:
                    if char in word:
                        num_chars += 1
                    matches[word] = num_chars
                features[word]['num_matches'] = matches[word] / (len(word) + 0.001) # for divide by zero error
                # except Exception as e:
                # print(str(e))
        return features


    def get_weights(self):
        """Custom weights to assign each feature based on validation set results"""
        weights = {
            'first_char_match':     0.63,
            'last_char_match':      0.62,
            'num_matches':          0.65,
            'exact_length_match':   0.43,
            'bin_length_match':     0.23,
            'levenshtein':          0.95,
            'editdistance':         0.49,
        }
        return weights
        

    def final_scores(self, features, ocr_pred, ocr_prob_threshold, return_topK=None):
        """
        Computes final score based on features and weights with some heuristics
        
        Args:
           features (dict): features for each model prediction token
           ocr_pred (list): OCR Predicted token
           ocr_prob_threshold (float): Probability threshold to return OCR model prediction
           return_topK (int): Return topK results 
        
        Returns:
            top_results (str): final predicted token
        """
        final_scores = {}

        for word, feature_dict in features.items():
            # if exact match in both LM and OCR model simply return word
            if feature_dict['exact'] or feature_dict['exact_stem'] or feature_dict['exact_lemma']:
                return word
            first_char_match = feature_dict['first_char_match']
            last_char_match = feature_dict['last_char_match']
            lm_prob = feature_dict['lm_prob']
            ocr_prob = feature_dict['ocr_prob']
            # if OCR model is really confident return OCR model prediction
            if ocr_prob >= ocr_prob_threshold:
                return ocr_pred
        # NEXT: if ocr prob is decently high, look for words with small edit distance in LM model
        weights = self.get_weights()
        # compute final score
        for word, dic in features.items():
            for feature in weights.keys():
                features[word].update({feature: (features[word][feature] * weights[feature])})
            final_scores[word] = sum(features[word].values())
        # sort top scores 
        top_results = sorted(final_scores.items(), key=itemgetter(1), reverse=True)
        if return_topK:
            return top_results[:return_topK]
        return top_results[0][0]

     
    def predict(self, sentence, img_path=None, ind_preds=None, ocr_prob_threshold=0.01, return_topK=None):
        """
        Computes final score based on features and weights with some heuristics
        
        Args:
           sentence (str): User Input string 
           img_path (str): Path to user uploaded image
           ind_preds (boolean): Return individual LM and OCR predictions  
           ocr_prob_threshold (float): Probability threshold to return OCR model prediction
           return_topK (int): Return topK results 
        
        Returns:
            out (str): final predicted token
        """
        # if valid image filepath and contains text
        valid_image = os.path.isfile(str(img_path))
        valid_text = False
        if re.search('[a-zA-Z&.,:;!?\d]', sentence) is not None:
            valid_text = True
        if valid_text:
            lm_preds = self.run_lm_inference_by_user_input(sentence)
        if valid_image:
            ocr_pred, ocr_pred_prob = self.run_beam_ocr_inference_by_user_image(img_path)
        if valid_text and valid_image:
            features = self.create_features_improved(lm_preds, ocr_pred, ocr_pred_prob)
            final_pred = self.final_scores(features, ocr_pred[0], ocr_prob_threshold)
            out = final_pred
            if ind_preds:
                out = final_pred, lm_preds[0], ocr_pred, ocr_pred_prob 
                if return_topK:
                    out = final_pred, lm_preds[:return_topK], ocr_pred, ocr_pred_prob 
        if not valid_image and not valid_text:
            return 'NO INPUT. TRY AGAIN'
        if not valid_image:
            out = lm_preds[0][1]
        if not valid_text:
            out = ocr_pred[0]
        
        return out