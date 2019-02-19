"""
This is for the N-Gram Model -- Not currently being used
"""

import numpy as np
import pandas as pd
import dill as pickle
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from collections import defaultdict
import shutil


# TODO remove this 
import warnings
warnings.filterwarnings("ignore")


def unpickle(filename):
    """ Unpickle file """
    with open(filename, 'rb') as f:
        return pickle.load(f)



DATA_DIR = '../../data'
data_path = os.path.join(DATA_DIR, 'raw/word_level')
meta_json_data_path = os.path.join(DATA_DIR, 'preprocessed/meta.json')
word_level_meta_path = os.path.join(DATA_DIR, 'preprocessed/word_level_meta.csv')
word_path_mapping_path = os.path.join(DATA_DIR, 'processed/word_path_mapping.pkl')
X_path = os.path.join(DATA_DIR, 'processed/X.npy')
y_path = os.path.join(DATA_DIR, 'processed/y.npy')
bigram_model_path = os.path.join(DATA_DIR, 'processed/ngram_models/bigram_likelihood_model.pkl')
trigram_model_path = os.path.join(DATA_DIR, 'processed/ngram_models/trigram_likelihood_model.pkl')
letters_path = os.path.join(DATA_DIR, 'processed/letters_map.pkl')


import boto3
from io import BytesIO
import dill as pickle



def s3_init(bucketname='handwrittingdetection'):

    session = boto3.Session(
        aws_access_key_id=pub_key,
        aws_secret_access_key=secret_key,
    )  
    client = session.resource('s3')
    bucket = client.Bucket(bucketname)
    return client, bucket

    
def unpickle_s3(filename, client=None, bucket=None):
    with BytesIO() as data:
        bucket.download_fileobj(filename, data)
        data.seek(0)
        return pickle.load(data)


def tokenize_and_join(context):
    tokens = word_tokenize(context)
    context = ' '.join(tokens)
    return tokens, context

def ngram_backoff_model(left_text, right_text, trigram_model, bigram_model, OOV_token=0):


    left_tokens, left_text = tokenize_and_join(left_text)
    right_tokens, right_text = tokenize_and_join(right_text)
    full_text = left_text + ' [] '  + right_text

    # get previous word(s)
    try:
        prev_word = left_tokens[-1]
    except:
        prev_word = '<bos>'
    try:
        prev_prev_word = left_tokens[-2]
    except:
        prev_prev_word = '<bos>'
    # model preds 
    print(prev_prev_word, prev_word)
    pred = trigram_model[(prev_prev_word, prev_word)]
    print('pred1', pred)
    if pred == OOV_token:
        pred = bigram_model[(prev_word)]
        print('pred2', pred)
        if pred == OOV_token:
            pred = 'UNK'
    return pred 


def teseract_baseline(file_url, word_path_mapping, letters, tmpdir='tmp/'):
    img_width = 256
    img_height = 100
    # gets image from sample index
    # img = word_path_mapping[y[sample_index][0]]
    im = Image.open(file_url)  # img is the path of the image
    im = im.convert("RGBA")
    im = im.resize((img_width, img_height))
    newimdata = []
    datas = im.getdata()

    vals = 255

    for item in datas:
        if item[0] < vals or item[1] < vals or item[2] < vals:
            newimdata.append(item)
        else:
            newimdata.append((255, 255, 255))
    im.putdata(newimdata)

    im = im.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert('1')
    
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    save_img_path = os.path.join(tmpdir, 'temp_img.jpg')
    im.save(save_img_path)
    
    text = pytesseract.image_to_string(Image.open(save_img_path),
                config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz -psm 13', lang='eng')
    
    return text, len(text) 


    
def get_ocr_model_pred(file_url, word_path_mapping, letters):
    ocr_pred, len_pred = teseract_baseline(file_url, word_path_mapping, letters)
    return ocr_pred, len_pred

def get_language_model_pred(left_text, right_text, trigram_model, bigram_model):
    pred = ngram_backoff_model(left_text, right_text, trigram_model, bigram_model)
    return pred


def get_pos_tags(left_text, right_text):
    return []

def weight_features(left_text, right_text, file_url, trigram_model, bigram_model, word_path_mapping, letters, weights={}):
    ocr_pred, len_pred = get_ocr_model_pred(file_url, word_path_mapping, letters)
    print('OCR', ocr_pred, len_pred)
    lm_pred = get_language_model_pred(left_text, right_text, trigram_model, bigram_model)
    print('LM', lm_pred)
    pos_pred = get_pos_tags(left_text, right_text)
    
    return ocr_pred, len_pred, lm_pred




def absoluteFilePaths(directory):
    """Walk filepaths"""
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.join(dirpath, f)


def create_image_path(df, data_path, use_s3=False, s3_image_path='data/word_level'):
    """Create dictionary for mapping of word to data path"""
    if use_s3:
        files = list(bucket.objects.filter(Prefix='data/word_level/sample'))
        all_paths = [f.key for f in files if '.png' in f.key]
    else:
        all_paths = [i for i in absoluteFilePaths(data_path)]
        
    all_path_endings = [i.split('/')[-1].split('.')[0] for i in all_paths]
    defaultdict(lambda: 0, dict(zip(all_path_endings, all_paths)))
    df['image_path'] = df['image_name'].map(lambda x: all_path_dict[x])
    return df



def get_prediction(left_text, right_text, file_url, use_s3=False):

    if use_s3:
        client, bucket = s3_init(bucketname='handwrittingdetection')

        bigram_model_path = 'data/ngram_models/bigram_likelihood_model.pkl'
        trigram_model_path = 'data/ngram_models/trigram_likelihood_model.pkl'

        word_path_mapping_path = 'data/word_path_mapping.pkl'
        letters_path = 'data/letters_map.pkl'

        bigram_model = unpickle_s3(bigram_model_path, client, bucket)
        trigram_model = unpickle_s3(trigram_model_path, client, bucket)

        word_path_mapping = unpickle_s3(word_path_mapping_path, client, bucket)
        letters = unpickle_s3(letters_path, client, bucket)


    else:
        data_path = os.path.join(DATA_DIR, 'raw/word_level')
        meta_json_data_path = os.path.join(DATA_DIR, 'preprocessed/meta.json')
        word_level_meta_path = os.path.join(DATA_DIR, 'preprocessed/word_level_meta.csv')
        word_path_mapping_path = os.path.join(DATA_DIR, 'processed/word_path_mapping.pkl')
        bigram_model_path = os.path.join(DATA_DIR, 'processed/ngram_models/bigram_likelihood_model.pkl')
        trigram_model_path = os.path.join(DATA_DIR, 'processed/ngram_models/trigram_likelihood_model.pkl')
        letters_path = os.path.join(DATA_DIR, 'processed/letters_map.pkl')

        # Language Models
        bigram_model = unpickle(bigram_model_path)
        trigram_model = unpickle(trigram_model_path)

        # Meta
        meta = pd.read_json(meta_json_data_path)
        word_level_df = pd.read_csv(word_level_meta_path)
        word_level_df = create_image_path(word_level_df, data_path)

        word_path_mapping = unpickle(word_path_mapping_path)
        letter2idx = unpickle(letters_path)

    letters = list(letter2idx.keys())
    letters = ''.join(letters[1:-2])

    left_text = left_text.rstrip()
    right_text = right_text.rstrip()

    ocr_pred, len_pred, lm_pred = weight_features(left_text, right_text, file_url, trigram_model, bigram_model, word_path_mapping, letters)
    return ocr_pred, len_pred, lm_pred





# if __name__ == '__main__':
#     sample_index = 102
#     main(sample_index)