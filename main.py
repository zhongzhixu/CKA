import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)

import tensorflow as tf
tf.reset_default_graph()
# default settings for Book-Crossing
DIMM = 4
N_MEMORY = 16
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='', help='which dataset to use')
parser.add_argument('--dim', type=int, default=DIMM, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--n_epoch', type=int, default=2, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=N_MEMORY, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--dimhidden', type=int, default=DIMM, help='size of hidden neurons of rnn')
parser.add_argument('--diminput', type=int, default=DIMM, help='the input dimension of rnn unit')
parser.add_argument('--dimoutput', type=int, default=DIMM, help='the output dimension of rnn unit')
parser.add_argument('--nsteps', type=int, default=3, help='number of time window')
parser.add_argument('--is_sequencial', type=bool, default=True, help='the sequence information is considered or not')
args = parser.parse_args()

show_loss = True
data_info = load_data(args)
train(args, data_info, show_loss)


