import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import time
import csv
from model import *
from generate import generate
from torch.utils import data
from torch.utils.data import DataLoader
from qa_dataset import qa_dataset,build_vocab
import pickle


# TODO
# download glove 100/300 d model
# initilize your model's embedding with glove 100d with available words in your vocabulary
# re-train the RNN to check if the performance is improved


if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--filename', type=str)
    argparser.add_argument('--model', type=str, default="gru", choices=['gru','lstm'])
    argparser.add_argument('--n_epochs', type=int, default=100)
    argparser.add_argument('--print_every', type=int, default=1)
    argparser.add_argument('--hidden_size', type=int, default=128) 
    argparser.add_argument('--n_layers', type=int, default=2)
    argparser.add_argument('--learning_rate', type=float, default=0.006)
    argparser.add_argument('--batch_size', type=int, default=8)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('-m', '--model_path', type=str, default='saves/rnn.pt')

    args = argparser.parse_args()

    main(args)
