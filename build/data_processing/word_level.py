from collections import defaultdict
import pandas as pd
import numpy as np
from glob import glob
import re
import os
from sklearn.model_selection import train_test_split


def duplicate_row(df, col_name):
    """
    When cell contents are lists, create a row for each element in the list

    Args: 
        df (pandas dataframe): DataFrame
        col_name (str): Column name 

    Returns:
        series (pandas series): Series with duplicate rows for each element
    """
    series = df.apply(lambda x: pd.Series(x[col_name]),axis=1).stack().reset_index(level=1, drop=True)
    series.name = col_name
    return series
    
def create_word_level_df(df, cols=[]):
    """
    Combine multiple Series into a pandas DataFrame

    Args:
        df (pandas dataframe): DataFrame to duplicate rows for

    Returns: 
        df (pandas dataframe): Concatinated pandas dataframe with token and images
    """
    meta_series = duplicate_row(df, 'meta')
    id_series = duplicate_row(df, 'ids')
    pos_series = duplicate_row(df, 'pos_tag')
    df = df.drop(['meta', 'ids', 'pos_tag'], axis=1).join(pd.concat([meta_series, id_series, pos_series], axis=1))
    return df.rename(columns={'meta': 'token', 'ids': 'image_name'}).reset_index()

def absoluteFilePaths(directory):
    """Walk filepaths"""
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.join(dirpath, f)

def create_image_path(df, data_path):
    """
    Create dictionary for mapping of word to data path

    Args:
        df (pandas dataframe): DataFrame to create mapping from 
        data_path (str): Path to dataset

    Returns:
        df (pandas dataframe): DataFrame with word to data path mapping
    """
    all_paths = [i for i in absoluteFilePaths(data_path)]
    all_path_endings = [i.split('/')[-1].split('.')[0] for i in all_paths]
    all_path_dict = defaultdict(lambda: 0, dict(zip(all_path_endings, all_paths)))
    df['image_path'] = df['image_name'].map(lambda x: all_path_dict[x])
    return df


def main_word_level(meta_json_data_path, image_data_path, test_size=0.2, save=False, 
                                        save_path='../../data/preprocessed'):
    """
    Reads, creates word level dataframes and saves to save path
    Calls create_word_level_df(), create_image_path()

    Args:
        meta_json_data_path (str): Path to meta dataset
        image_data_path (str): Path to image data
        test_size (float): Testing set percentage
        save (boolean): Save dataframe
        save_path (str): Directory to save data

    Returns:
        None
    """
    print('Reading Meta')
    meta = pd.read_json(meta_json_data_path)
    meta.drop(index=494, axis=1, inplace=True)
    print('Create word level df')
    word_level_df = create_word_level_df(meta)
    print('Create Image paths')
    word_level_df = create_image_path(word_level_df, image_data_path)
    # split to train / test 
    print('Split train and test for test size of {}'.format(test_size))
    train, test = train_test_split(word_level_df, test_size=test_size, random_state=100)
    if save:
        print('Saving data to {}'.format(save_path))
        train.to_csv(os.path.join(save_path, 'word_level_train.csv'), index=False)
        test.to_csv(os.path.join(save_path, 'word_level_test.csv'), index=False)
    
    
if __name__ == '__main__':
    image_data_path = '../../data/raw/word_level'
    meta_json_data_path = '../../data/preprocessed/meta.json'    
    main_word_level(meta_json_data_path, image_data_path, save=True, test_size=0.2)