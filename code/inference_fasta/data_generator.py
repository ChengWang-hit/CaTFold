import os

import _pickle as cPickle
from torch.utils import data
from utils import *
import torch
from multiprocessing import Pool

class DataGenerator(object):
	def __init__(self, data_dir, node_input_dim):
		self.data_dir = data_dir
		self.node_input_dim = node_input_dim
		self.load_data()

	def load_data(self):
		names, sequences = read_fasta(self.data_dir)
		if len(sequences) == 0:
			print('No sequences found in the fasta file.')
			return

		self.seqs = sequences
		self.name = names
		self.len = len(self.seqs)

	def get_one_sample(self, index):
		seq = self.seqs[index]
		node_onehot = seq2onehot(seq)
		seq_length = len(seq)

		# sinusoidal position embedding
		node_pe = absolute_position_embedding(seq_length, self.node_input_dim)

		# Create a mask for valid base pairs (A-U, G-C, G-U)
		mask_matrix = create_mask_matrix(seq)

		# bipartite
		node_set1 = seq2set(seq)

		result = (
			torch.Tensor(node_onehot), \
			torch.Tensor(node_pe), \
			torch.Tensor(mask_matrix), \
			seq_length, \
			node_set1)
		
		return result

class Dataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, data):
		'Initialization'
		self.data = data

	def __len__(self):
		'Denotes the total number of samples'
		return self.data.len

	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		return self.data.get_one_sample(index)