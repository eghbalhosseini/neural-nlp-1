#small adjustments to lmzoo script (added BOS token)

#inspiration https://github.com/HendrikStrobelt/detecting-fake-text/blob/master/backend/api.py
#(GLTR: Giant Language Model Test Room)

#Accompanying Colab notebook for tests: https://colab.research.google.com/drive/1ANRNKcp_WpxY8R5f-eCjrYK0mYFzKX_e#scrollTo=EdpuzHlTK-4x
"""
Get surprisal estimates for a transformers model.
"""

import argparse
import os
import logging
import operator
from pathlib import Path
import sys

import h5py
import torch
import numpy as np

import torch
from scipy.special import softmax
from transformers import GPT2Tokenizer, GPT2LMHeadModel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)


def readlines(inputf):
    with inputf as f:
        lines = f.readlines()
    lines = [l.strip('\n') for l in lines]
    return lines

def set_seed(seed, cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def _get_predictions_inner(sentence, tokenizer, model): #, device):

    sentence = tokenizer.bos_token + " " + sentence #prepend beginning-of-sentence token (e.g. https://github.com/huggingface/transformers/issues/1009)

    sent_tokens = tokenizer.tokenize(sentence)
    print(sent_tokens)
    indexed_tokens = tokenizer.convert_tokens_to_ids(sent_tokens)
    print(indexed_tokens)
    # create 1 * T input token tensor
    tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)
    #tokens_tensor = tokens_tensor.to(device)

    #The output of GPT2LMHeadModel are logits so you can just apply a softmax (or log-softmax for log probabilities) on them to get probabilities for each token. 
    #Then if you want to get probabilities for words, you will need to multiply (or add if you used a log-softmax) the probabilities of the sub-words in each word.
    with torch.no_grad():
        logits = model(tokens_tensor)[0]
        log_probs = logits.log_softmax(dim=-1).squeeze() #log_softmaxes logits so we can just add surprisals rather than multiply for a word.

    #print(len(log_probs), len(log_probs[0]), len(indexed_tokens))

    # None for BOS token
    return list(zip(sent_tokens, indexed_tokens, (None,) + log_probs.unbind())) #None as first word doesn't have any context.


def get_predictions(sentence, tokenizer, model): #, device):
    for token, idx, probs in _get_predictions_inner(sentence, tokenizer, model): #, device):
        yield token, idx, probs.numpy() if probs is not None else probs


def get_surprisals(sentence, tokenizer, model): #, device):
    predictions = _get_predictions_inner(sentence, tokenizer, model) #, device)
    #print(predictions)

    surprisals = []
    for word, word_idx, preds in predictions:
        print(word, word_idx, preds)
        if type(preds) == type(None): #Solution from here: https://github.com/pytorch/pytorch/issues/5486#issuecomment-501934308
        #if preds == None:
            surprisal = 0.0
        else:
            surprisal = -preds[word_idx].item() / np.log(2) # this turns log_probabilities to surprisal in bits. (-log_e(p)/log_e(2) == -log_2(p))

        surprisals.append((word, word_idx, surprisal))

    return surprisals

#We summed the surprisal values of every word in each sentence, as in Wilcox et al. (2018),


def main(args):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    #device = torch.device('cuda')
    #model.to(device)
    model.eval()
    
    logger.info('Reading sentences from %s...', args.inputf)
    sentences = readlines(args.inputf)
   

    with args.outputf as f:
        f.write("sentence_id\ttoken_id\ttoken\tsurprisal\n")
        
        for i, sentence in enumerate(sentences):
            print(sentence)
            surprisals = get_surprisals(sentence, tokenizer, model) #, device)
            # write surprisals for sentence (append to outputf)
            for j, (word, word_idx, surprisal) in enumerate(surprisals):
                f.write("%i\t%i\t%s\t%f\n" % (i + 1, j + 1, word, surprisal))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get token-level model surprisal estimates')
    parser.add_argument("--inputf", type=argparse.FileType("r", encoding="utf-8"),
                        help="Input file")
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--outputf', '-o', type=argparse.FileType("w"), default=sys.stdout,
                        help='output file for generated text')
    main(parser.parse_args())