import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import pearsonr
from scipy.special import softmax
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.base import clone
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import plotly.express as px
import pickle
import string
from transformers import GPT2Tokenizer, GPT2Model
# from nltk.corpus import words
import copy
import warnings
import re
import os
import timeit
from pathlib import Path


def flatten_activations(activations):
    """
    Convert activations into dataframe format
    Input: dict, key = layer, item = 2D array of stimuli x units
    Output: pd dataframe, stimulus ID x MultiIndex (layer, unit)
    """
    labels = []
    arr_flat = []
    for layer, arr in activations.items():
        arr = np.array(arr)
        arr_flat.append(arr)
        for i in range(arr.shape[1]):
            labels.append((layer, i))
    arr_flat = np.concatenate(arr_flat, axis=1)
    df = pd.DataFrame(arr_flat)
    df.columns = pd.MultiIndex.from_tuples(labels)
    return df

def flatten_activations_arr(activations):
    """
    Convert activations into dataframe format
    Input: 3D array layer x stimuli x units
    Output: pd dataframe, stimulus ID x MultiIndex (layer, unit)
    """
    d = {}
    for i in range(activations.shape[0]):
        d[i] = activations[i]
    return flatten_activations(d)

def get_activations(model, tokenizer, sents, sentence_embedding, lower=True, verbose=True):
    """
    :param sents: list of strings
    :param sentence_embedding: string denoting how to obtain sentence embedding
    Compute activations of hidden units
    Returns dict with key = layer, item = 2D array of stimuli x units
    """
    
    model.eval() # does not make a difference
    n_layer = model.config.n_layer
    max_n_tokens = model.config.n_positions
    states_sentences = defaultdict(list)
    

    if verbose:
        print(f'Computing activations for {len(sents)} sentences')

    for count, sent in enumerate(sents):
        if lower:
            sent = sent.lower()
        input_ids = torch.tensor(tokenizer.encode(sent))

        print(tokenizer.convert_ids_to_tokens(input_ids))
        
        
        start = max(0, len(input_ids) - max_n_tokens)
        if start > 0:
            warnings.warn(f"Stimulus too long! Truncated the first {start} tokens")
        input_ids = input_ids[start:]
        result_model = model(input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = result_model['hidden_states']
        for i in range(n_layer+1): # for each layer
            if sentence_embedding == 'last-tok':
                state = hidden_states[i][-1,:].detach().numpy() # get last token
            elif sentence_embedding == 'avg-tok':
                state = torch.mean(hidden_states[i], dim=0).detach().numpy() # mean over tokens
            elif sentence_embedding == 'sum-tok':
                state = torch.sum(hidden_states[i], dim=0).detach().numpy() # sum over tokens
            else:
                print('Sentence embedding method not implemented')
            states_sentences[i].append(state)
            
    return states_sentences
