from utils import *
import warnings

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

from transformers import GPT2Tokenizer, GPT2Model, BertTokenizer, BertModel
from neural_nlp.models import LM1B

import os
import getpass
import sys
import datetime
from scipy.spatial import distance
import pickle

from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
import torch
import numpy as np
from tqdm import tqdm
import pickle
import os
import argparse
import pandas as pd

import logging
logger = logging.getLogger(__name__)

class GPT2():
    def __init__(self, model_instance, sentence_index_zip=None, sentence_embedding='avg-tok'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_instance, add_prefix_space=True)
        self.model = GPT2Model.from_pretrained(model_instance, output_hidden_states=True, return_dict=True) #return_dict=True returns output as dict and not plain tuple.
        self.name = model_instance
        self.sentence_index_zip = sentence_index_zip
        self.sentence_embedding = sentence_embedding

    def _encode(self, sentence_index_zip):
      """input: pickle file with index and sentence columns
      output: dictionary mapping from sentence index to tensor of all hidden states!
      e.g., for BERT, there are 13 layers á number of words in the sentence (here: 8) á hidden_dim (here: 768),
      first argument is batch size (here: 1):
      print(np.shape(outputs['hidden_states'])) #(13,)
      print(np.shape(outputs['hidden_states'][0])) #torch.Size([1, 8, 768]) > use .squeeze() to get rid of
      dim1 dimension >> torch.Size([8, 768])
      """
      reps_dict = {}
      token_dict = {}

      #encode sentences
      for (index, sent) in tqdm(sentence_index_zip):
          print(index)
          inputs = self.tokenizer(sent, return_tensors="pt")
          token_dict[index] = np.sort(inputs['input_ids'])
          outputs = self.model(**inputs)
          print(outputs.keys())
          token_reps = outputs['hidden_states'] #index, e.g. with token_reps[-1] for last layer representation

          layers = ['embedding'] + [f'hidden.layer.{i}' for i in range(self.model.config.n_layer)]

          sentence_embedding_dict = {}
          for ind, layer in enumerate(layers):
              sentence_embedding_dict[layer] = {}
              token_reps_layer = token_reps[ind]
              if ind == 0:
                print(f'Shape of model output: {np.shape(token_reps_layer)}')
              token_reps_layer = token_reps_layer.squeeze()
              if self.sentence_embedding == 'last-tok':
                state = token_reps_layer[-1,:].detach().numpy() # get last token
                if ind == 0:
                  print(f'Shape of model output after squeeze: {np.shape(token_reps_layer)}')
                  print(f'Shape after sentence embedding method: {np.shape(state)}')
              elif self.sentence_embedding == 'avg-tok':
                state = torch.mean(token_reps_layer, dim=0).detach().numpy() # mean over tokens
                if ind == 0:
                  print(f'Shape of model output after squeeze: {np.shape(token_reps_layer)}')
                  print(f'Shape after sentence embedding method: {np.shape(state)}')
              elif self.sentence_embedding == 'sum-tok':
                  state = torch.sum(token_reps_layer, dim=0).detach().numpy() # sum over tokens
              else:
                  print('Sentence embedding method not implemented')
              sentence_embedding_dict[layer][index] = state
      return sentence_embedding_dict, token_dict

class Bert():
    def __init__(self, model_instance, sentence_index_zip=None, sentence_embedding='avg-tok',exclude_spec_tokens=False):
        self.tokenizer = BertTokenizer.from_pretrained(model_instance)
        self.model = BertModel.from_pretrained(model_instance, output_hidden_states=True, return_dict=True) #return_dict=True returns output as dict and not plain tuple.
        self.name = model_instance
        self.sentence_index_zip = sentence_index_zip
        self.sentence_embedding = sentence_embedding
        self.exclude_spec_tokens = exclude_spec_tokens

    def _encode(self, sentence_index_zip):
      """input: pickle file with index and sentence columns
      output: dictionary mapping from sentence index to tensor of all hidden states!
      e.g., for BERT, there are 13 layers á number of words in the sentence (here: 8) á hidden_dim (here: 768),
      first argument is batch size (here: 1):
      print(np.shape(outputs['hidden_states'])) #(13,)
      print(np.shape(outputs['hidden_states'][0])) #torch.Size([1, 8, 768]) > use .squeeze() to get rid of
      dim1 dimension >> torch.Size([8, 768])
      """
      reps_dict = {}
      token_dict = {}

      #encode sentences
      for (index, sent) in tqdm(sentence_index_zip):
          print(index)
          inputs = self.tokenizer(sent, return_tensors="pt")
          token_dict[index] = np.sort(inputs['input_ids'])
          outputs = self.model(**inputs)
          print(outputs.keys())
          token_reps = outputs['hidden_states'] #index, e.g. with token_reps[-1] for last layer representation

          layers = ['embedding'] + [f'hidden.layer.{i}' for i in range(self.model.config.num_hidden_layers)]

          sentence_embedding_dict = {}
          for ind, layer in enumerate(layers):
              sentence_embedding_dict[layer] = {}
              token_reps_layer = token_reps[ind]
              if ind == 0:
                print(f'Shape of model output: {np.shape(token_reps_layer)}')
              token_reps_layer = token_reps_layer.squeeze()

              if self.exclude_spec_tokens == True:
                  token_reps_layer = token_reps_layer[1:-1,:] # exclude [CLS], [SEP]

              if self.sentence_embedding == 'last-tok':
                state = token_reps_layer[-1,:].detach().numpy() # get last token
                if ind == 0:
                  print(f'Shape of model output after squeeze: {np.shape(token_reps_layer)}')
                  print(f'Shape after sentence embedding method: {np.shape(state)}')
              elif self.sentence_embedding == 'avg-tok':
                state = torch.mean(token_reps_layer, dim=0).detach().numpy() # mean over tokens
                if ind == 0:
                  print(f'Shape of model output after squeeze: {np.shape(token_reps_layer)}')
                  print(f'Shape after sentence embedding method: {np.shape(state)}')
              elif self.sentence_embedding == 'sum-tok':
                  state = torch.sum(token_reps_layer, dim=0).detach().numpy() # sum over tokens
              else:
                  print('Sentence embedding method not implemented')
              sentence_embedding_dict[layer][index] = state
      return sentence_embedding_dict, token_dict

def preprocess_stimuli(sentences, final_period=None):
    if final_period:
        sentences = [elm.rstrip().lower() + '.' for elm in sentences]
    else:
        sentences = [elm.rstrip().lower() for elm in sentences]
    print(f'Example of preprocessed sentence: {sentences[0]}')
    return sentences


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description='scrambled sentence reps. parser')
    parser.add_argument('--sentence_embedding', type=str) #default is 'avg-tok'
    parser.add_argument('--final_period', type=bool) #True or False
    parser.add_argument('--model', type=str)
    parser.add_argument('--scrambled_version', type=str)
    args = parser.parse_args()
    
    logger.info("***** STARTING! *****")
    
    #load data
    scrambled_data_dir = "/om/user/ckauf/neural-nlp/ressources/scrambled-stimuli-dfs/"

    STIMULI_TO_PKL_MAP = {'lowPMI': os.path.join(scrambled_data_dir, 'stimuli_lowPMI.pkl'),
                      'Original': os.path.join(scrambled_data_dir, 'stimuli_Original.pkl'),
                      'Scr1': os.path.join(scrambled_data_dir, 'stimuli_Scr1.pkl'),
                      'Scr3': os.path.join(scrambled_data_dir, 'stimuli_Scr3.pkl'),
                      'Scr5': os.path.join(scrambled_data_dir, 'stimuli_Scr5.pkl'),
                      'Scr7': os.path.join(scrambled_data_dir, 'stimuli_Scr7.pkl'),
                      'random': os.path.join(scrambled_data_dir, 'stimuli_random.pkl'),
                      'backward': os.path.join(scrambled_data_dir, 'stimuli_backward.pkl')    
                      }

    for key in STIMULI_TO_PKL_MAP.keys():
        if args.scrambled_version == key:
            print(f"I AM USING THIS DATA VERSION: {key}")
            x = pd.read_pickle(STIMULI_TO_PKL_MAP[key])
            print(f"I AM SAVING A NEW PICKLE FILE FOR {args.scrambled_version}!")
            x.to_pickle('stimuli_indep_{}.pkl'.format(args.scrambled_version)) #reminder: remove after 1 run

            stimulus_ids = list(x['stimulus_id'])
            sentences = list(x['sentence'])
            prep_sentences = preprocess_stimuli(sentences, final_period=args.final_period)
            sentence_index_zip = list(zip(stimulus_ids, prep_sentences))
    
    if args.model.startswith("bert"):
        model = Bert(args.model, sentence_index_zip=None, sentence_embedding=args.sentence_embedding, exclude_spec_tokens=False)
    elif args.model.startswith("gpt2"):
        model = GPT2(args.model, sentence_embedding=args.sentence_embedding)
    #elif args.model.startswith("lm1b"):
    #    model = LM1B
    else:
        warnings.warn("model doesn't exist")
        
    sentence_embedding_dict, token_dict = model._encode(sentence_index_zip)
    
    # Directory
    directory = "{}".format(args.model)  # e.g. "gpt2"

    # Save Directory path
    save_dir = "/om/user/ckauf/neural-nlp/neural_nlp/analyze/neural-scrambled/metric-validation/model-activations"

    # Path
    path = os.path.join(save_dir, directory)
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory '%s' created" % directory)
    
    fname_act = os.path.join(path,'activations_model=_{}_{}_{}_{}_Pereira2018.pkl'.format(model.name, args.scrambled_version, args.sentence_embedding, str(args.final_period)))
    with open(fname_act, 'wb') as fout_act:
        pickle.dump(sentence_embedding_dict, fout_act)

    fname_tok = os.path.join(path,'tokens_model=_{}_{}_{}_{}_Pereira2018.pkl'.format(model.name, args.scrambled_version, args.sentence_embedding, str(args.final_period)))
    with open(fname_tok, 'wb') as fout_tok:
        pickle.dump(token_dict, fout_tok)