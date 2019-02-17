# LM model imports 
import os
import time
import torch
from torch import optim
# from models.context2vec.src.eval.mscc import mscc_evaluation
# from models.context2vec.src.core.nets import Context2vec
# from models.context2vec.src.util.args import parse_args
# from models.context2vec.src.util.batch import Dataset
# from models.context2vec.src.util.config import Config
# from models.context2vec.src.util.io import write_embedding, write_config, read_config, load_vocab

# from src.mscc_eval import mscc_evaluation
# from src.model import Context2vec
# from src.negative_sampling import NegativeSampling
# from src.utils import write_embedding, write_config 
# from src.dataset import WikiDataset
# import boto3
# from io import BytesIO
# from src.args import parse_args
# from src.config import Config


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
from models.OCRBeamSearch.src.DataLoader import DataLoader, Batch


from collections import defaultdict
from operator import itemgetter
from nltk.stem import PorterStemmer, WordNetLemmatizer
import Levenshtein
import re


# tf.reset_default_graph()

class FilePaths:
    "filenames and paths to data"
    fnCharList = 'models/OCRBeamSearch/model/charList.txt'
    fnAccuracy = 'models/OCRBeamSearch/model/accuracy.txt'
    fnTrain = 'models/OCRBeamSearch/data/'
    fnInfer = 'models/OCRBeamSearch/data/test.png'
    fnCorpus = 'models/OCRBeamSearch/data/corpus.txt'



class Inference():
    def __init__(self, modelfile=None, wordsfile=None, img_width=128, img_height=64, device='cpu'):
        
        self.device = device
        self.img_width = img_width
        self.img_height = img_height
        self.build_language_model()
#         self.build_ocr_model()
        self.build_beam_ocr_model()
    
        self.stemmer = PorterStemmer()
        self.lemma = WordNetLemmatizer()
        
    def build_language_model(self, model_dir='models/context2vec/models_103'):
        # LANGUAGE MODEL
        modelfile = os.path.join(model_dir, 'model.param')
        wordsfile = os.path.join(model_dir, 'embedding.vec')
#         modelfile = 'models/context2vec/models/model.param'
#         wordsfile = 'models/context2vec/models/embedding.vec'
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
#         optimizer = optim.Adam(model.parameters(), lr=config_dict['learning_rate'])
        # optimizer.load_state_dict(torch.load(modelfile+'.optim'))
        self.itos, self.stoi = load_vocab(wordsfile)
        self.unk_token = config_dict['unk_token']
        self.bos_token = config_dict['bos_token']
        self.eos_token = config_dict['eos_token']


    def build_beam_ocr_model(self, decoderType = 'wordbeamsearch'):
        # Beam Search OCR model
        decoderType = 'wordbeamsearch'
        if 'beamsearch':
            decoderType = DecoderType.BeamSearch
        if 'wordbeamsearch':
            decoderType = DecoderType.WordBeamSearch
        self.beam_ocr_model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
               
    def build_ocr_model(self):
        self.sess = tf.Session()
        K.set_session(self.sess)

        ocr_model_path = 'models/ocr/models/weights-improvement2-10-01-3.00.hdf5'
        self.ocr_model = load_model(ocr_model_path, custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
        
    def preprocess_image(self, img_path, img_width, img_height):
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
        bad_list = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<unk>']
        # evaluation mode 
        self.lm_model.eval()
        # norm_weight
        self.lm_model.norm_embedding_weight(self.lm_model.criterion.W)

        tokens, target_pos = self._return_split_sentence(sentence)
        tokens[target_pos] = self.unk_token
        tokens = [self.bos_token] + tokens + [self.eos_token]
        indexed_sentence = [self.stoi[token] if token in self.stoi else self.stoi[self.unk_token] for token in tokens]
        input_tokens = \
            torch.tensor(indexed_sentence, dtype=torch.long, device=self.device).unsqueeze(0)
        topv, topi = self.lm_model.run_inference(input_tokens, target=None, target_pos=target_pos, k=topK)
        output = []  
        for value, key in zip(topv, topi):
            word = self.itos[key.item()]
            if word not in bad_list:
                output.append((value.item(), word))
#             print(value.item(), self.itos[key.item()])
        return output

    def run_beam_ocr_inference_by_user_image(self, img_path):
        "recognize text in image provided by file path"
        img = preprocess(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), Model.imgSize)
        batch = Batch(None, [img])
        (recognized, probability) = self.beam_ocr_model.inferBatch(batch, True)
#         print('Recognized:', '"' + recognized[0] + '"')
#         print('Probability:', probability[0])
        return (recognized, probability)
    
    def run_ocr_inference_by_user_image(self, img):
        net_inp = self.ocr_model.get_layer(name='the_input').input
        net_out = self.ocr_model.get_layer(name='softmax').output
        net_out_value = self.sess.run(net_out, feed_dict={net_inp: img})
        pred_texts = self._decode_batch(net_out_value)
        return pred_texts


    def create_features(self, lm_preds, ocr_pred):

        # not used currently
        # ----------------------------------
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
#         ocr_len = len([x for x in ocr_pred])
        pred_bins = [k for k, v in bins.items() if ocr_len in v]
        # ----------------------------------

        features = {}
        bad_list = []#['<PAD>', '<BOS>', '<EOS>', '<UNK>'] # ADD foul words
        matches = {}
        matches_non_ordered = {}

        ocr_pred_lower = ocr_pred[0].lower()
#         ocr_pred_lower = ocr_pred.lower()

        for lm_pred in lm_preds:
            score, word = lm_pred[0], lm_pred[1].rstrip()
            word = word.lower()
            # remove pad, bos, etc...
            if word not in bad_list:
                try:
                    features[word] = {}
                    features[word]['score'] = score
                    # match first and last character 
                    first_char_match = word[0] == ocr_pred_lower[0]
                    last_char_match = word[-1] == ocr_pred_lower[-1]
                    features[word]['first_char_match'] = first_char_match
                    features[word]['last_char_match'] = last_char_match

                    num_chars = 0
                    for char in ocr_pred_lower:
                        if char in word:
                            num_chars += 1
                        matches[word] = num_chars
                    features[word]['num_matches'] = matches[word] 
                except:
                    pass 

        return features

    def create_features_improved(self, lm_preds, ocr_pred, ocr_prob):

        # not used currently
        # ----------------------------------
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
#         ocr_len = len([x for x in ocr_pred])
        ocr_pred_bins = [k for k, v in bins.items() if ocr_len in v]
#         print(ocr_pred_bins)
        # ----------------------------------

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



                num_chars = 0
                for char in ocr_pred_lower:
                    if char in word:
                        num_chars += 1
                    matches[word] = num_chars
                features[word]['num_matches'] = matches[word] / (len(word) + 0.001) # for divide by zero error
#                 except Exception as e:
#                     print(str(e))
#                     pass 
#         from pprint import pprint
#         pprint(features)
        return features

    def get_weights(self):
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
        final_scores = {}

        for word, feature_dict in features.items():
#             if exact match in both LM and OCR model simply return word
            if feature_dict['exact'] or feature_dict['exact_stem'] or feature_dict['exact_lemma']:
#                 print('Perfect Match', word)
                return word
#             final_score = 1
            first_char_match = feature_dict['first_char_match']
            last_char_match = feature_dict['last_char_match']
            lm_prob = feature_dict['lm_prob']
            ocr_prob = feature_dict['ocr_prob']
            # if OCR model is really confident return OCR model prediction
            if ocr_prob >= ocr_prob_threshold:
#                 print('ocr_prob_threshold')
                return ocr_pred
            
#             if first_char_match:
#                 final_score += 10
#             if last_char_match:
#                 final_score += 10
#             final_score *= score

#             final_scores[word] = final_score
        weights = self.get_weights()
    

        for word, dic in features.items():
            for feature in weights.keys():
                features[word].update({feature: (features[word][feature] * weights[feature])})
            final_scores[word] = sum(features[word].values())


        top_results = sorted(final_scores.items(), key=itemgetter(1), reverse=True)
        if return_topK:
            return top_results[:return_topK]
        return top_results[0][0]

    def weigh_function(self):
        pass
    
    
    def predict(self, sentence, img_path=None, ind_preds=None, ocr_prob_threshold=0.01, return_topK=None):
        
        # if valid image filepath and contains text
        valid_image = os.path.isfile(img_path)
        valid_text = False
#         if re.search('[a-zA-Z]', sentence) is not None:
        if re.search('[a-zA-Z&.,:;!?\d]', sentence) is not None:
            valid_text = True
#         print(valid_text, valid_image)
        if valid_text:
            lm_preds = self.run_lm_inference_by_user_input(sentence)
        if valid_image:
            ocr_pred, ocr_pred_prob = self.run_beam_ocr_inference_by_user_image(img_path)
    
#         features = self.create_features(lm_preds, ocr_pred, ocr_pred_prob)
        if valid_text and valid_image:
            features = self.create_features_improved(lm_preds, ocr_pred, ocr_pred_prob)
#             print("LM", lm_preds[0])
#             print("OCR", ocr_pred)
#             print("OCR Prob", ocr_pred_prob)
            final_pred = self.final_scores(features, ocr_pred[0], ocr_prob_threshold)
            out = final_pred
            if ind_preds:
#                 print('both')
                out = final_pred, lm_preds[0], ocr_pred, ocr_pred_prob 
                if return_topK:
                    out = final_pred, lm_preds[:return_topK], ocr_pred, ocr_pred_prob 
        # return top K? 
        if not valid_image and not valid_text:
            return 'NO INPUT. TRY AGAIN'
        if not valid_image:
#             print('text only')
            out = lm_preds[0]
        if not valid_text:
#             print('image only')
            out = ocr_pred
        
        return out


# if __name__ == '__main__':
#     left_text = 'the dog ran'
#     right_text = 'the house'

#     sentence = left_text + ' [] ' + right_text
#     img_path = '../../data/samples/c03-096f-03-05.png'


#     inference = Inference()

#     print(inference.predict(sentence, img_path))