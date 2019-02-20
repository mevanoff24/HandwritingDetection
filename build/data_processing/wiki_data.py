import pandas as pd
import numpy as np
import codecs
from nltk.tokenize import sent_tokenize
from collections import Counter, defaultdict
import re
import dill as pickle
import random
import os

def create_full_path(wiki_path, dataset):
    """Joins to paths together for wiki dataset"""
    return os.path.join(wiki_path, 'wiki.{}.tokens'.format(dataset))

def create_data(wiki_path, dataset, available_image_letters, save=False, target_sep=['<<', '>>'], 
                threshold_length=3, iteration_threshold=30, out_path='../../data/processed/'):
    """
    Loads and searchs for useable target word (one we have an image for) and saves wiki text dataset
    
    Args:
        wiki_path (str): Path to wiki dataset
        dataset (str): Dataset type (train, valid, test)
        available_image_letters (list): All letters to be used
        save (boolean): Save dataframe
        target_sep (list): Tokens used to surround target variable
        threshold_length (int): Only use target above this word length
        iteration_threshold (int): Iterations to run before passing on sentence
        out_path (str): Initial folder to save data
    
    Returns:
        None 
    """
    dataset_name = wiki_path.split('/')[-1]
    data_path = create_full_path(wikitext2_path, 'train')
    
    random.seed(100)
    X = []
    y = []
    raw = []
    bad_words = ['=', '戦', '場', 'の', 'ヴ', 'ァ', 'ル', 'キ', 'ュ', 'リ', 'ア', '戦場のヴァルキュリア3']
    N = 0
    bad_lines = 0
    counter = Counter()
    
    with codecs.open(data_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.rstrip():
                if not any(bad_word in line for bad_word in bad_words):
                    for sent in sent_tokenize(line):
                        split_sent = sent.split()
                        found = True
                        iteration = 0
                        while (found == True):
                            iteration += 1
                            random_int = random.randint(0, len(split_sent)-1)
                            target = split_sent[random_int]
                            if (len(target) >= threshold_length) and (target in available_image_letters):
                                X.append(split_sent[:random_int] + [''.join([target_sep[0]] + [target] + [target_sep[1]])] 
                                             + split_sent[random_int+1:])
                                y.append((target, random_int))
                                raw.append(split_sent)
                                found = False
                                # get word counts 
                                for token in split_sent:
                                    counter[token] += 1

                            if (iteration >= iteration_threshold):
                                found = False
                                bad_lines += 1
                if N % 5000 == 0:
                    print('processed {} lines'.format(N))
                # if N >= 100: break
                N += 1
        print('number of skipped lines', bad_lines)
    if save:
        print('Save Path', out_path + '----' + dataset_name + '-' + dataset)
        np.save(out_path + 'X_' + dataset_name + '-' + dataset, X)
        np.save(out_path + 'y_' + dataset_name + '-' + dataset, y)
        np.save(out_path + 'raw' + dataset_name + '-' + dataset, raw)



def main_wiki(wikitext_path, dataset, save=False):
    """
    Run and save Wiki data

    Args:
        wikitext_path (str): Path to wiki dataset
        dataset (str): Dataset type (train, valid, test)
        save (boolean): Save dataframe

    Returns:
        None
    """
    print('Reading word level meta from {}'.format(dataset))
    word_level_meta_path = '../../data/preprocessed/word_level_{}.csv'.format(dataset)
    word_level_df = pd.read_csv(word_level_meta_path)
    available_image_letters = word_level_df.token.values.tolist()
    print('first 10 available image letters: ', available_image_letters[:10])
    print('Building Dataset')
    create_data(wikitext_path, dataset, available_image_letters, save=save)

if __name__ == '__main__':
    wikitext2_path = '../../data/raw/language_model/wikitext-2'
    wikitext103_path = '../../data/raw/language_model/wikitext-103'
    word_level_meta_path_all = '../../data/preprocessed/word_level_meta.csv'
    word_level_meta_path_train = '../../data/preprocessed/word_level_train.csv'
    word_level_meta_path_test = '../../data/preprocessed/word_level_test.csv'
    main_wiki(wikitext103_path, dataset='train', save=True)
#     main_wiki(wikitext2_path, dataset='test', save=True)
    