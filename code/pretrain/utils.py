import argparse
import os
import random
import networkx as nx
from scipy import signal

import numpy as np
import torch
import sys

class Logger(object):
	def __init__(self, fileN='Default.log'):
		self.terminal = sys.stdout
		sys.stdout = self
		self.log = open(fileN, 'w')

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def reset(self):
		self.log.close()
		sys.stdout=self.terminal
	
	def flush(self):
		pass

# IUPAC
onehot_dict = {
	'A':[1,0,0,0],
	'U':[0,1,0,0],
	'C':[0,0,1,0],
	'G':[0,0,0,1],
	'T':[0,1,0,0],
	'N':[1,1,1,1],
	'M':[1,0,1,0],
	'Y':[0,1,1,0],
	'W':[1,1,0,0],
	'V':[1,0,1,1],
	'K':[0,1,0,1],
	'R':[1,0,0,1],
	'X':[0,0,0,0],
	'S':[0,0,1,1],
	'D':[1,1,0,1],
	'B':[0,1,1,1],
	'H':[1,1,1,0]
}

def evaluate(pred_a, true_a, eps=1e-11):
	tp = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a)).sum()
	pred_p = torch.sign(torch.Tensor(pred_a)).sum()
	true_p = true_a.sum()
	fp = pred_p - tp
	fn = true_p - tp
	recall = (tp + eps)/(tp+fn+eps)
	precision = (tp + eps)/(tp+fp+eps)
	f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)
	return precision.cpu(), recall.cpu(), f1_score.cpu()

def evaluate_shifted(pred_a, true_a, eps=1e-11):
	kernel = np.array([[0.0,1.0,0.0],
					   [1.0,1.0,1.0],
					   [0.0,1.0,0.0]])
	pred_a_filtered = signal.convolve2d(np.array(pred_a.cpu()), kernel, 'same')
	fn = len(torch.where((true_a - torch.Tensor(pred_a_filtered).cuda())==1)[0])
	pred_p = torch.sign(torch.Tensor(pred_a)).sum()
	true_p = true_a.sum()
	tp = true_p - fn
	fp = pred_p - tp
	recall_s = (tp + eps) / (tp + fn + eps)
	precision_s = (tp + eps) / (tp + fp + eps)
	f1_score_s = (2*tp + eps) / (2*tp + fp + fn + eps)
	return precision_s.cpu(), recall_s.cpu(), f1_score_s.cpu()

def get_args():
	argparser = argparse.ArgumentParser(description="diff through pp")
	argparser.add_argument(
		'-c', '--config',
		metavar='C',
		default='code/pretrain/config.json',
		help='The Configuration file'
	)
	argparser.add_argument(
		'-r', '--rank',
		metavar='C',
		default=0,
	)
	args = argparser.parse_args()
	return args

def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def post_process_argmax(p, threshold=0.5):
	max_values, max_indices = torch.max(p, dim=1)
	p = torch.where(p == max_values.view(-1, 1), p, torch.zeros_like(p))
	
	p = torch.where(p > threshold, torch.ones_like(p), torch.zeros_like(p))
	return p.cuda()

# hopcroft_karp algorithm
def post_process_HK(param):
	mat, node_set1, threshold = param

	mat = torch.where(mat > threshold, mat, torch.zeros_like(mat))
	n = mat.size(-1)
	G = nx.convert_matrix.from_numpy_array(np.array(mat.data.cpu()))
	# top_nodes = [v for i,v in enumerate(G.nodes) if bipartite_label[i] == 0]
	pairings = nx.bipartite.maximum_matching(G, top_nodes=node_set1)
	y_out = torch.zeros_like(mat)
	for (i, j) in pairings.items():
		if i>n and j>n:
			continue
		y_out[i%n, j%n] = 1
		y_out[j%n, i%n] = 1
	return y_out.cuda()

def seq2onehot(seq):
	onehot = []
	for s in seq:
		try:
			feature = onehot_dict[s]
		except:
			feature = [0,0,0,0]
		onehot.append(feature)
	return np.array(onehot)

def absolute_position_embedding(seq_length, embedding_dim):
	node_pe = torch.zeros(seq_length, embedding_dim, dtype=torch.float)
	position = torch.arange(0, seq_length).unsqueeze(dim=1).float()
	div_term = (10000 ** ((2*torch.arange(0, embedding_dim/2)) / embedding_dim)).unsqueeze(dim=1).T
	node_pe[:, 0::2] = torch.sin(position @ (1/div_term))
	node_pe[:, 1::2] = torch.cos(position @ (1/div_term))
	return node_pe

def pairs2map(pairs, seq_len, prob=None):
	contact = torch.zeros([seq_len, seq_len])
	idx = torch.LongTensor(pairs.astype(float)).transpose(0, 1)
	if prob is None:
		contact[idx[0], idx[1]] = 1
	else:
		contact[idx[0], idx[1]] = torch.Tensor(prob)
	return contact

def seq2set(seq):
	set1 = {'A', 'G'}
	node_set1 = []
	for i, s in enumerate(seq):
		if s in set1:
			node_set1.append(i)
	return node_set1

def dotbracket2pairs(ss):
	stack = []
	pairs = []
	for i in range(len(ss)):
		if ss[i] == '(':
			stack.append(i)
		
		elif ss[i] == ')':
			pairs.append([stack[-1], i])
			pairs.append([i, stack[-1]])
			stack.pop()
	
		else:
			continue
	return pairs

def collate(data_list):
	node_onehot_list = []
	node_pe_list = []
	pred_pairs_list = []
	pad_mask_list = []
	seq_length_list = [data[3] for data in data_list]
	max_length = max(seq_length_list)
	label_ss_list = []
 
	offset = 0
	for i in range(len(data_list)):
		node_onehot = torch.zeros((max_length, data_list[i][0].size(-1)), dtype=torch.float)
		node_onehot[:seq_length_list[i]] = data_list[i][0]
		node_onehot_list.append(node_onehot)

		node_pe = torch.zeros((max_length, data_list[i][1].size(-1)), dtype=torch.float)
		node_pe[:seq_length_list[i]] = data_list[i][1]
		node_pe_list.append(node_pe)

		pred_pairs_list.append(data_list[i][2] + offset)

		pad_mask = torch.zeros(max_length, dtype=torch.bool)
		pad_mask[seq_length_list[i]:] = True
		pad_mask_list.append(pad_mask)
  
		label_ss_list.append(data_list[i][4])

		offset += seq_length_list[i]

	return torch.stack(node_onehot_list), \
		   torch.stack(node_pe_list), \
		   torch.cat(pred_pairs_list, dim=1), \
		   torch.stack(pad_mask_list), \
		   torch.LongTensor(seq_length_list), \
		   torch.cat(label_ss_list, dim=0)
		   
def collate_test(data_list):
	node_onehot_list = []
	node_pe_list = []
	pred_pairs_list = []
	pred_pair_num_list = []
	pad_mask_list = []
	contact_map_list = []
	seq_length_list = [data[5] for data in data_list]
	max_length = max(seq_length_list)
	node_set1_list = []
 
	offset = 0
	for i in range(len(data_list)):
		node_onehot = torch.zeros((max_length, data_list[i][0].size(-1)), dtype=torch.float)
		node_onehot[:seq_length_list[i]] = data_list[i][0]
		node_onehot_list.append(node_onehot)

		node_pe = torch.zeros((max_length, data_list[i][1].size(-1)), dtype=torch.float)
		node_pe[:seq_length_list[i]] = data_list[i][1]
		node_pe_list.append(node_pe)

		pred_pairs_list.append(data_list[i][2] + offset)

		pad_mask = torch.zeros(max_length, dtype=torch.bool)
		pad_mask[seq_length_list[i]:] = True
		pad_mask_list.append(pad_mask)
  
		pred_pair_num_list.append(data_list[i][3])
  
		contact_map_list.append(data_list[i][4])
  
		node_set1_list.append(data_list[i][6])

		offset += seq_length_list[i]

	return torch.stack(node_onehot_list), \
		   torch.stack(node_pe_list), \
		   torch.cat(pred_pairs_list, dim=1), \
           torch.LongTensor(pred_pair_num_list), \
		   torch.stack(pad_mask_list), \
           torch.cat(contact_map_list, dim=0), \
		   torch.LongTensor(seq_length_list), \
           node_set1_list
         