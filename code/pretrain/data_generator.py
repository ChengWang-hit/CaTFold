import os

import _pickle as cPickle
from torch.utils import data
from utils import *
import torch
from multiprocessing import Pool
from tqdm import tqdm
import h5py

class DataGenerator(object):
	def __init__(self, data_dir, split, node_input_dim, mode='pretrain'):
		self.data_dir = data_dir
		self.split = split
		self.node_input_dim = node_input_dim
		self.mode = mode
		if self.mode == 'family_fold':
			self.mask_dir = self.data_dir.replace('/family_fold', '')
		else:
			self.mask_dir = self.data_dir
		self.load_data()

	def load_data(self):
		data_dir = self.data_dir
		
		test_num = int(1e9) # full
		# test_num = int(1000000) # test
		if self.mode == 'pretrain':
			print(f"Loading pretrain data...")
			with h5py.File(os.path.join(data_dir, f'{self.split}.h5'), 'r') as f:
				self.data = f
				self.seq = [seq.decode('utf-8').upper() for seq in tqdm(self.data['seq'][:test_num])]
				self.ss = [ss.decode('utf-8').upper() for ss in tqdm(self.data['ss'][:test_num])]
			self.len = len(self.seq)
		else:
			with open(os.path.join(data_dir, f'{self.split}.pickle'), 'rb') as f:
				self.data = cPickle.load(f)
			self.seq = [seq.upper() for seq in tqdm(self.data['seq'][:test_num])]
			self.ss = self.data['ss'][:test_num]
			self.mask_matrix_idx = self.data['mask_matrix_idx'][:test_num]
			self.len = len(self.seq)

	def get_one_sample(self, index):
		seq = self.seq[index]
		node_onehot_original = seq2onehot(seq)
		seq_length = len(seq)
		
		# sinusoidal position embedding
		node_pe = absolute_position_embedding(seq_length, self.node_input_dim)

		if self.mode == 'pretrain':
			# ss
			cm_pairs = np.array(dotbracket2pairs(self.ss[index])) # (N, 2)
			if len(cm_pairs) == 0:
				cm_pairs = np.array([[0, seq_length-1], [seq_length-1, 0]])
			contact_map = pairs2map(cm_pairs, seq_length)
			row_indices, col_indices = torch.triu_indices(seq_length, seq_length, offset=1) # Upper triangle
			pred_pairs = torch.vstack((row_indices, col_indices))
			pred_pair_num = pred_pairs.shape[1]
			label_ss = contact_map[pred_pairs[0, :], pred_pairs[1, :]]

			# mutation
			prob = 0.02
			sequence_mask = torch.rand(seq_length) < prob # single nucleotide
			if sequence_mask.sum() < 1:
				sequence_mask[random.randint(0, seq_length-1)] = True
			index_mut = torch.where(sequence_mask)[0]

			node_onehot_mut = node_onehot_original.copy()
			for idx in index_mut:
				zeros_vec = np.zeros(4)
				random_nuc_idx = random.randint(0, 3) 
				zeros_vec[random_nuc_idx] = 1
				node_onehot_mut[idx] = zeros_vec

			result = (
					torch.Tensor(node_onehot_mut), \
					torch.Tensor(node_pe), \
					
					torch.LongTensor(pred_pairs), \
															
					seq_length, \
					
					torch.Tensor(label_ss))
		else:
			with open(os.path.join(self.mask_dir, 'mask_matrix', f'{self.mask_matrix_idx[index]}.pickle'), 'rb') as f:
				pred_pairs = torch.LongTensor(cPickle.load(f)).T # (2, N)
			pred_pair_num = pred_pairs.shape[1]		

			cm_pairs = np.array(self.ss[index])
			if len(cm_pairs) == 0:
				cm_pairs = np.array([[0, seq_length-1], [seq_length-1, 0]])
			contact_map = pairs2map(cm_pairs, seq_length)

			# bipartite graph
			node_set1 = seq2set(seq)

			result = (
       			torch.Tensor(node_onehot_original), \
				torch.Tensor(node_pe), \
				torch.LongTensor(pred_pairs), \
				pred_pair_num, \
				torch.Tensor(contact_map).view(-1), \
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