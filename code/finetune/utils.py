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

char_dict = {
	0: 'A',
	1: 'U',
	2: 'C',
	3: 'G'
}

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
	# tp = (preds * labels).sum((1,2))
	# pred_p = preds.sum((1,2))
	# true_p = labels.sum((1,2))

	tp = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a)).sum()
	pred_p = torch.sign(torch.Tensor(pred_a)).sum()
	true_p = true_a.sum()
	fp = pred_p - tp
	fn = true_p - tp
	recall = (tp + eps)/(tp+fn+eps)
	precision = (tp + eps)/(tp+fp+eps)
	f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)
	return precision.item(), recall.item(), f1_score.item()

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
		default='code/finetune/config.json',
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

def post_process_maximum_weight_matching(param):
	mat, node_set1, threshold = param

	mat_np = mat.data.cpu().numpy()

	mat_np[mat_np <= threshold] = 0
	G = nx.convert_matrix.from_numpy_array(mat_np)

	pairings_set = nx.max_weight_matching(G, maxcardinality=False, weight='weight')

	y_out = torch.zeros_like(mat)
	n = mat.size(-1)

	for u, v in pairings_set:
		if 0 <= u < n and 0 <= v < n:
			y_out[u, v] = 1
			y_out[v, u] = 1
   
	return y_out.cuda()

def seq2onehot(seq):
	onehot = []
	for s in seq:
		# feature = onehot_dict[s]
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

def pairs2map(pairs, seq_len):
    if np.array(pairs).size == 0:
        return torch.zeros([seq_len, seq_len])
    contact = torch.zeros([seq_len, seq_len])
    idx = torch.LongTensor(pairs).T
    contact[idx[0], idx[1]] = 1
    return contact

def seq2set(seq):
	set1 = {'A', 'G'}
	node_set1 = []
	for i, s in enumerate(seq):
		if s in set1:
			node_set1.append(i)
	return node_set1

def outer_concat(t1, t2):
	seq_len = t1.shape[1]
	a = t1.unsqueeze(-2).expand(-1, -1, seq_len, -1)
	b = t2.unsqueeze(-3).expand(-1, seq_len, -1, -1)

	return torch.concat((a, b), dim=-1)

def collate(data_list):
	node_onehot_list = []
	node_pe_list = []
	pad_mask_list = []
	seq_length_list = [data[2] for data in data_list]
	max_length = max(seq_length_list)
	label_ss_list = []
 
	for i in range(len(data_list)):
		node_onehot = torch.zeros((max_length, data_list[i][0].size(-1)), dtype=torch.float)
		node_onehot[:seq_length_list[i]] = data_list[i][0]
		node_onehot_list.append(node_onehot)

		node_pe = torch.zeros((max_length, data_list[i][1].size(-1)), dtype=torch.float)
		node_pe[:seq_length_list[i]] = data_list[i][1]
		node_pe_list.append(node_pe)

		pad_mask = torch.zeros(max_length, dtype=torch.bool)
		pad_mask[seq_length_list[i]:] = True
		pad_mask_list.append(pad_mask)
  
		label_ss_list.append(data_list[i][3])
  
	return torch.stack(node_onehot_list), \
		   torch.stack(node_pe_list), \
		   torch.stack(pad_mask_list), \
		   torch.LongTensor(seq_length_list), \
		   torch.cat(label_ss_list, dim=0)
			
def collate_test(data_list):
	node_onehot_list = []
	node_pe_list = []
	pad_mask_list = []
	mask_matrix_list = []
	contact_map_list = []
	seq_length_list = [data[4] for data in data_list]
	max_length = max(seq_length_list)
	node_set1_list = []
 
	for i in range(len(data_list)):
		node_onehot = torch.zeros((max_length, data_list[i][0].size(-1)), dtype=torch.float)
		node_onehot[:seq_length_list[i]] = data_list[i][0]
		node_onehot_list.append(node_onehot)

		node_pe = torch.zeros((max_length, data_list[i][1].size(-1)), dtype=torch.float)
		node_pe[:seq_length_list[i]] = data_list[i][1]
		node_pe_list.append(node_pe)

		pad_mask = torch.zeros(max_length, dtype=torch.bool)
		pad_mask[seq_length_list[i]:] = True
		pad_mask_list.append(pad_mask)
  
		mask_matrix = torch.zeros((max_length, max_length), dtype=torch.float)
		mask_matrix[:seq_length_list[i], :seq_length_list[i]] = data_list[i][2]
		mask_matrix_list.append(mask_matrix)
		
		contact_map = torch.zeros((max_length, max_length), dtype=torch.float)
		contact_map[:seq_length_list[i], :seq_length_list[i]] = data_list[i][3]
		contact_map_list.append(contact_map)
  
		node_set1_list.append(data_list[i][5])

	return torch.stack(node_onehot_list), \
		   torch.stack(node_pe_list), \
		   torch.stack(pad_mask_list), \
		   torch.stack(mask_matrix_list), \
		   torch.stack(contact_map_list), \
		   torch.LongTensor(seq_length_list), \
		   node_set1_list