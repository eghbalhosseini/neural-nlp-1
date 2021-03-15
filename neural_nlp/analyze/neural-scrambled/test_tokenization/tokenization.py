from utils import *

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from transformers import GPT2Tokenizer, GPT2Model
import os
import getpass
import sys
import datetime
from scipy.spatial import distance
import pickle
import gzip
DATADIR = (Path(os.getcwd()) / '..' / 'data').resolve()
from scipy import stats

if __name__ == '__main__':
	model_info = {"gpt2": (GPT2Model.from_pretrained('gpt2'),
						   (GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True))),
				  "gpt2-xl": (GPT2Model.from_pretrained('gpt2-xl'),
							  (GPT2Tokenizer.from_pretrained('gpt2-xl')))}
	
	# Load source model
	source_model, source_tokenizer = model_info['gpt2']
	
	s1 = 'beekeeping encourages the conservation of local habitats'
	s2 = 'conservation of beekeeping encourages the habitats local'
	s3 = 'the quick brown fox jumps over a lazy dog'
	
	s = [s1,s2,s3]
	
	activations1 = get_activations(model=source_model, tokenizer=source_tokenizer,
								  sents=s, sentence_embedding='avg-tok',
								  verbose=True)  # compute activations
	activations1 = flatten_activations(activations1)[12].to_numpy()  # convert into a single dataframe
	
	distance.euclidean(activations1[0], activations1[1])
	np.corrcoef(activations1[0],activations1[2])
	
	## CKA

	# os.chdir('/Users/gt/Documents/GitHub/CKA-Centered-Kernel-Alignment')

	import cca_core
	from CKA import linear_CKA, kernel_CKA
	
	# load pereira stim
	# Load data: pereira
	stimuli_pereira = pd.read_csv(os.path.join(DATADIR, 'pereira_stimulus_set.csv'))
	stimuli_pereira = stimuli_pereira.set_index('stimulus_id')
	stimuli_pereira = stimuli_pereira.sentence.values
	
	stimuli_random = []
	for s in stimuli_pereira:
		sc = scramble_words(s)
		stimuli_random.append(sc)
		
	activations = get_activations(model=source_model, tokenizer=source_tokenizer,
								  sents=stimuli_pereira, sentence_embedding='avg-tok',
								  verbose=True)  # compute activations
	activations = flatten_activations(activations)[12].to_numpy()
	
	activations_rand = get_activations(model=source_model, tokenizer=source_tokenizer,
								  sents=stimuli_random, sentence_embedding='avg-tok',
								  verbose=True)  # compute activations
	activations_rand = flatten_activations(activations_rand)[12].to_numpy()
	
	# CKA linear
	lCKA = linear_CKA(activations, activations_rand)
	
	# CKA kernel
	kCKA = kernel_CKA(activations, activations_rand)
	
	# Pearson R
	pearsonR = np.corrcoef(activations.flatten(), activations_rand.flatten())
	
	# Spearman
	spearmanR = stats.spearmanr(activations.flatten(), activations_rand.flatten())
	
	# Euc
	all_euc = []
	for i, e in enumerate(range(0, activations.shape[0])):
		euc = distance.euclidean(activations[i], activations_rand[i])
		all_euc.append(euc)
		
	# Cos
	all_cos = []
	for i, e in enumerate(range(0, activations.shape[0])):
		cos = distance.cosine(activations[i], activations_rand[i])
		all_cos.append(cos)
	
	all_euc_mean = np.mean(all_euc)
	all_cos_mean = np.mean(all_cos)
	
	# clip
	activations_clip = np.clip(activations, -1, 1)
	activations_clip_rand = np.clip(activations_rand, -1, 1)
	
	# CKA linear
	lCKA = linear_CKA(activations_clip, activations_clip_rand)
	
	# CKA kernel
	kCKA = kernel_CKA(activations_clip, activations_clip_rand)
	
	# Pearson R
	pearsonR = np.corrcoef(activations_clip.flatten(), activations_clip_rand.flatten())
	
	# Spearman
	spearmanR = stats.spearmanr(activations_clip.flatten(), activations_clip_rand.flatten())
	
	# Euc
	all_euc = []
	for i, e in enumerate(range(0, activations_clip.shape[0])):
		euc = distance.euclidean(activations_clip[i], activations_clip_rand[i])
		all_euc.append(euc)
	
	# Cos
	all_cos = []
	for i, e in enumerate(range(0, activations_clip.shape[0])):
		cos = distance.cosine(activations_clip[i], activations_clip_rand[i])
		all_cos.append(cos)
	
	all_euc_mean = np.mean(all_euc)
	all_cos_mean = np.mean(all_cos)