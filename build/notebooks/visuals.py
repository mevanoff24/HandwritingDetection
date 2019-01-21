from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_x_y(meta, sample_idx, data_path):
    """
    Plots image and label for sanity checks
    Args:
        meta (pandas dataframe): Meta data
        sample_idx (int): Row from meta dataframe
        data_path (str): Path to meta data
    """
    sample_meta = meta.loc[sample_idx]
    sample_path = os.path.join(data_path, sample_meta.folder, sample_meta.filename)
    individual_file_paths = sorted(glob(sample_path + '/*'))
    print(sample_meta.meta)
    print(len(sample_meta.meta), len(individual_file_paths))
    plt.figure(figsize=(100, 500))
    for i in range(min(len(sample_meta.meta), len(individual_file_paths))):
        plt.subplot(50, 2, i+1)
        plt.imshow(plt.imread(individual_file_paths[i]), cmap='gray')
        plt.title(sample_meta.meta[i], fontdict = {'fontsize' : 100})

        
        
def plot_most_common(counts, topN=20):
    """
    Bar plot based on most common counts
    Args:
        counts (Counter obect): Counts of token and frequency
        topN (int): Top number of most common occurances to plot  
    """
    x, y = zip(*counts.most_common(topN))
    plt.bar(x, y);
    
    