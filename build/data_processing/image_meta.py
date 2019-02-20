import pandas as pd
import numpy as np
from glob import glob
import os
import re
import xml.etree.ElementTree as ET

def get_structed_data(data_path):
    """
    Parse XML file to extract relevant information for model

    Args:
        data_path (str): Path to data

    Returns: 
        all_data (dict): Dictionary of filename with text, ids, and pos tag
    """
    all_data = {}
    for filename in glob(data_path):
        tree = ET.parse(filename)
        root = tree.getroot()
        tmp = []
        ids = []
        pos_tag = []
        writer_id = root.attrib['writer-id']
            
        for part in root.findall('handwritten-part'):
            for line in part.findall('line'):
                for word in line.findall('word'):
                    tmp.append(word.attrib['text'].rstrip())
                    ids.append(word.attrib['id'].rstrip())
                    pos_tag.append(word.attrib['tag'].rstrip())
        assert(len(tmp) == len(ids) == len(pos_tag))
        all_data[filename.split('/')[-1].split('.')[0]] = [tmp, ids, pos_tag]
    return all_data


def create_dataframe(all_data):
    """
    Create dataframe from dictionary with additional features
    folder, meta, document, 'ids', 'pos_tag'

    Args:
        all_data (dict): Dictionary of filename with text, ids, and pos tag 

    Returns: 
        dat (pandas dataframe): DataFrame with features of folder, meta, and document
    """
    dat = pd.DataFrame(all_data).T.reset_index()
    dat.columns = ['filename', 'meta', 'ids', 'pos_tag']
    dat['folder'] = dat.filename.map(lambda x: x.split('-')[0])
    dat['meta'] = dat.meta.map(lambda x: np.array([i.replace('&quot;', '"') for i in x]))
    dat['document'] = dat.filename.map(lambda x: re.sub(r'[A-Za-z]', '', x.split('-')[-1]))
    dat['document'] = dat['folder'] + '-' + dat['document']
    return dat


def main_meta(data_path, save=False, save_path='../../data/preprocessed'):
    """
    Parse XML file and creates dataframe. 
    Calls get_structed_data() and create_dataframe() functions. 

    Args:
        data_path (str): Path to data
        save (boolean): Save dataframe
        save_path (str): Path to save dataframe

    Returns: 
        None
    """
    print('Getting data')
    all_data = get_structed_data(data_path)
    print('Creating data')
    dat = create_dataframe(all_data)
    # to CSV 
    if save:
        print('Saving data to {}'.format(save_path))
        dat.to_csv(os.path.join(save_path, 'meta.csv'), index=False)
        dat.to_json(os.path.join(save_path, 'meta.json'))
    
if __name__ == '__main__':
    data_path = '../../data/raw/xml/*'
    main_meta(data_path, save=False)