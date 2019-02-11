import numpy as np
import pandas as pd
import re
import time
from inference import Inference
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import pairwise_distances

# import tensorflow as tf
# tf.reset_default_graph()

def create_embeddings(file_path='models/embeddings/glove.6B.100d.txt'):
    embeddings = {}

    with open(file_path) as f:
        for line in f:
            line = line.split()
            word, vector = line[0], line[1:]
            embeddings[word] = np.asarray(vector, np.float32)
    return embeddings



def evaluate_model(X_test, y_test, word_level_df_test, subset=None, ind_preds=True, only_ocr=False, only_lm=False, 
                                               embedding_file='models/embeddings/glove.6B.100d.txt'):
    
    if only_ocr:
        print('ONLY EVALUATING OCR MODEL FOR {} SAMPLES'.format(subset))
    if only_lm:
        print('ONLY EVALUATING LM MODEL FOR {} SAMPLES'.format(subset))
    embeddings = create_embeddings(embedding_file)
    stemmer = PorterStemmer()
    lemma = WordNetLemmatizer()
    
    y_preds = []
    y_true = [y[0].lower() for y in y_test[:subset]]
    N = 0
    n_correct = 0
    n_correct_stem = 0
    total_similarity = 0
    bad_lines = 0

    
    len_test = len(X_test[:subset])
    start = time.time()
    for i, (X, y) in enumerate(zip(X_test[:subset], y_true)):
        try:
#         print(X)
            X = re.sub(r'<<([^;]*)>>', '[]', ' '.join(X))
        #     print(y)
            img_path_df = word_level_df_test.loc[word_level_df_test.token == y, 'image_path']
            if len(img_path_df) > 0:
                N += 1
                img_path = np.random.choice(img_path_df.values.tolist())
                # still getting 'unk'
    #                 y_pred = inference_model.predict(X, img_path)
                y_pred, lm_pred, ocr_pred, ocr_pred_prob = inference_model.predict(X, img_path, ind_preds=ind_preds, return_topK=50)
#                 print(y_pred, lm_pred, ocr_pred)
    #             print('y_pred', y_pred)
#                 print('lm_pred', lm_pred)
    #             print('ocr_pred', ocr_pred)
                if only_ocr:
                    y_pred = ocr_pred[0]
                if only_lm:
    #                 print(lm_pred)
                    y_pred = lm_pred[0]
    #             print('Predicted: ', y_pred)
    #             print('-----------')
                # exact match accuracy
                if y_pred.lower() == y:
                    n_correct += 1
                else:
                    print('final pred: "{}" -- ocr pred: "{}" {} -- lm pred: "{}" {} -- true: "{}"'.format(
                                y_pred.lower(), ocr_pred[0], round(ocr_pred_prob[0]*100,2), lm_pred[0][1], y in [w for _, w in lm_pred], y))
                # stemmed accuracy 
                if stemmer.stem(y_pred) == stemmer.stem(y):
                    n_correct_stem += 1
                converge_counter = 0
                try:
                    lemma_pred = lemma.lemmatize(y_pred)
                    lemma_true = lemma.lemmatize(y)
                    similarity = pairwise_distances([embeddings[lemma_pred]], [embeddings[lemma_true]])[0][0]
                    total_similarity += similarity
                    converge_counter += 1
                except:
                    # add a score of 10 if obsure word is present
                    total_similarity += 10

            else:
                bad_lines += 1
            if i % 100 == 0:
                print('processed {} %'.format((i / len_test)*100.0))
        except Exception as e:
            print(str(e))
            print('bad line')

    finish = time.time()
    raw_correct = n_correct / N
    stem_correct = n_correct_stem / N
    average_total_similarity = total_similarity / N
    
    
    print('Raw correct', raw_correct)
    print('Stem correct', stem_correct)
    print('Average Cosine Similarity {}. Coverage: {}'.format(average_total_similarity, round(converge_counter / N, 3)))
    
    print('Time to evaluate {} sample: {}'.format(subset, finish - start))
    print('Number of bad lines', bad_lines)
    
    return raw_correct, stem_correct, average_total_similarity


if __name__ == '__main__':

    data_path = '../../data/processed/'
#     X_train = np.load(data_path + 'X_wikitext-2-train.npy')
    X_test = np.load(data_path + 'X_wikitext-103-test.npy')
#     y_train = np.load(data_path + 'y_wikitext-2-train.npy')
    y_test = np.load(data_path + 'y_wikitext-103-test.npy')
    word_level_df_test = pd.read_csv('../../data/preprocessed/word_level_test.csv')

    # init class 
    inference_model = Inference(device='cpu')
    # evaluate model 
    raw_correct, stem_correct, average_total_similarity = evaluate_model(X_test, y_test, 
                                                    word_level_df_test, subset=10000, 
                                                ind_preds=True, only_ocr=False, only_lm=False)
    
    
    
    

# ONLY OCR
# Raw correct 0.9086967610303494
# Stem correct 0.9120122417750574
# Average Cosine Similarity 0.6765553216851239. Coverage: 0.0
# Time to evaluate 10000 sample: 5024.438623189926
# Number of bad lines 2158


# ONLY LM
# Raw correct 0.25975516449885233
# Stem correct 0.2626880897730171
# Average Cosine Similarity 4.91581974263811. Coverage: 0.0
# Time to evaluate 10000 sample: 5024.891078233719
# Number of bad lines 2158

# Combined

# Raw correct 0.9108645753634277
# Stem correct 0.916475388931395
# Average Cosine Similarity 0.6160382933089897. Coverage: 0.0
# Time to evaluate 10000 sample: 5070.504088401794
# Number of bad lines 2158


