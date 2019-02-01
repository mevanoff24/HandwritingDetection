import numpy as np
import pandas as pd
import re
import time
from inference import Inference
from nltk.stem import PorterStemmer




def evaluate_model(X_test, y_test, word_level_df_test, subset=None):
    
    stemmer = PorterStemmer()
    
    y_preds = []
    y_true = [y[0].lower() for y in y_test[:subset]]
    N = 0
    n_correct = 0
    n_correct_stem = 0
    bad_lines = 0


    len_test = len(X_test[:subset])
    start = time.time()
    for i, (X, y) in enumerate(zip(X_test[:subset], y_true)):
        try:
            X = re.sub(r'<<([^;]*)>>', '[]', ' '.join(X))
        #     print(y)
            img_path_df = word_level_df_test.loc[word_level_df_test.token == y, 'image_path']
            if len(img_path_df) > 0:
                N += 1
                img_path = np.random.choice(img_path_df.values.tolist())
                # still getting 'unk'
                y_pred = inference_model.predict(X, img_path)
                if y_pred.lower() == y:
                    n_correct += 1
                if stemmer.stem(y_pred) == stemmer.stem(y):
                    n_correct_stem += 1

            else:
                bad_lines += 1
            if i % 100 == 0:
                print('processed {} %'.format((i / len_test)*100.0))
        except:
            print('bad line')

    finish = time.time()
    raw_correct = n_correct / N
    stem_correct = n_correct_stem / N
    
    print('Raw correct', raw_correct)
    print('Stem correct', stem_correct)
    print('Time to evaluate {} sample: {}'.format(subset, finish - start))
    print('Number of bad lines', bad_lines)
    
    return raw_correct, stem_correct


if __name__ == '__main__':

    data_path = '../../data/processed/'
#     X_train = np.load(data_path + 'X_wikitext-2-train.npy')
    X_test = np.load(data_path + 'X_wikitext-2-test.npy')
#     y_train = np.load(data_path + 'y_wikitext-2-train.npy')
    y_test = np.load(data_path + 'y_wikitext-2-test.npy')
    word_level_df_test = pd.read_csv('../../data/preprocessed/word_level_test.csv')

    # init class 
    inference_model = Inference(img_width=128, img_height=64, device='cpu')
    # evaluate model 
    raw_correct, stem_correct = evaluate_model(X_test, y_test, word_level_df_test, subset=1000)