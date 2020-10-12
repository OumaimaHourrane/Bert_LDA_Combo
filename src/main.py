from sklearn.externals import joblib
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse

if __name__ == '__main__':

    sentences = joblib.load('sentences.pkl')
    token_lists = joblib.load('token_lists.pkl')
    idx_in = joblib.load('idx_in.pkl')
    with open('LDA_2020_05_19_18_01_23.file', 'rb') as f:
        tm = pickle.load(f)    

    print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
    print('Silhouette Score:', get_silhouette(tm))
