import argparse
import networkx as nx
from Bio import SeqIO
import forgi
import forgi.visual.mplotlib as fvm
import forgi.graph.bulge_graph as fgb
import matplotlib.pyplot as plt

import numpy as np
import torch
import sys
import os
from tqdm import tqdm

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

def get_args():
	argparser = argparse.ArgumentParser(description="diff through pp")
	argparser.add_argument(
		'-c', '--config',
		metavar='C',
		default='code/inference_fasta/config.json',
		help='The Configuration file'
	)
	argparser.add_argument(
		'-r', '--rank',
		metavar='C',
		default=0,
	)
	args = argparser.parse_args()
	return args

# hopcroft_karp algorithm
def post_process_HK(param):
	mat, node_set1, threshold = param

	thresholded_mask = mat > threshold
	thresholded_mat = np.where(thresholded_mask, mat, np.zeros_like(mat))
	n = thresholded_mat.shape[-1]
	G = nx.convert_matrix.from_numpy_array(thresholded_mat)
	# top_nodes = [v for i,v in enumerate(G.nodes) if bipartite_label[i] == 0]
	pairings = nx.bipartite.maximum_matching(G, top_nodes=node_set1)
	y_out = np.zeros((n, n), dtype=int)
	for (u, v) in pairings.items():
		if 0 <= u < n and 0 <= v < n:
			y_out[u, v] = 1
			y_out[v, u] = 1
	return y_out

def post_process_maximum_weight_matching(param):
	mat, node_set1, threshold = param

	mat_np = mat

	mat_np[mat_np <= threshold] = 0

	G = nx.convert_matrix.from_numpy_array(mat_np)

	pairings_set = nx.max_weight_matching(G, maxcardinality=False, weight='weight')

	y_out = np.zeros_like(mat)
	n = mat.shape[-1]

	for u, v in pairings_set:
		if 0 <= u < n and 0 <= v < n:
			y_out[u, v] = 1
			y_out[v, u] = 1
   
	return y_out

def read_fasta(filepath):
	names = []
	sequences = []
	loaded_count = 0

	# Define the set of allowed RNA bases
	# allowed_bases = set("AUGC")

	print(f"Attempting to load RNA sequences from file: '{filepath}'...")

	try:
		for record in SeqIO.parse(filepath, "fasta"):
			raw_seq_str = str(record.seq).upper()
			rna_seq_str = raw_seq_str.replace('T', 'U')

			names.append(record.id) # record.id is the sequence identifier extracted by Biopython
			sequences.append(rna_seq_str)
			loaded_count += 1
	
		print(f"File '{filepath}' processing complete.")
		print(f"Successfully loaded {loaded_count} sequences meeting the criteria.")
		
	except Exception as e:
		print(f"An error occurred while reading or parsing the file with Biopython: {e}")
		return [], [] # Return empty lists for other errors

	return names, sequences

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

def create_mask_matrix(sequence, min_loop_length=4):
    L = len(sequence)

    if L == 0:
        return np.zeros((0, 0), dtype=float)

    sequence = sequence.upper().replace('T', 'U')
    seq_array = np.array(list(sequence))

    seq_row = seq_array.reshape(L, 1)
    seq_col = seq_array.reshape(1, L)

    mask_AU = (seq_row == 'A') & (seq_col == 'U')
    mask_UA = (seq_row == 'U') & (seq_col == 'A')
    mask_GC = (seq_row == 'G') & (seq_col == 'C')
    mask_CG = (seq_row == 'C') & (seq_col == 'G')
    mask_GU = (seq_row == 'G') & (seq_col == 'U')
    mask_UG = (seq_row == 'U') & (seq_col == 'G')

    pairing_mask = mask_AU | mask_UA | mask_GC | mask_CG | mask_GU | mask_UG

    indices = np.arange(L)
    dist_matrix = indices.reshape(1, L) - indices.reshape(L, 1)
    
    loop_constraint_mask = np.abs(dist_matrix) >= min_loop_length
    
    final_mask = pairing_mask & loop_constraint_mask

    pairing_matrix = final_mask.astype(float)

    return pairing_matrix

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
	mask_matrix_list = []
	pad_mask_list = []
	seq_length_list = [data[3] for data in data_list]
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
	
		node_set1_list.append(data_list[i][4])

	return torch.stack(node_onehot_list), \
		   torch.stack(node_pe_list), \
		   torch.stack(pad_mask_list), \
		   torch.stack(mask_matrix_list, dim=0), \
		   torch.LongTensor(seq_length_list), \
		   node_set1_list

def contact_map_to_bpseq(sequence, contact_map):
	"""Converts a sequence and its contact map to the BPSEQ format string."""
	seq_len = len(sequence)
	pairing = np.zeros(seq_len, dtype=int)
	
	paired_indices = np.where(contact_map)
	
	for i, j in zip(*paired_indices):
		if i < j:
			pairing[i] = j + 1
			pairing[j] = i + 1
			
	bpseq_lines = []
	for i in range(seq_len):
		line = f"{i+1} {sequence[i]} {int(pairing[i])}"
		bpseq_lines.append(line)
		
	return "\n".join(bpseq_lines)

def pairs_to_dot_bracket(pairs, seq_len):
	structure = ['.'] * seq_len
	for i, j in pairs:
		structure[i] = '('
		structure[j] = ')'
	return "".join(structure)

def contact_map_to_pairs(contact_map):
    if not isinstance(contact_map, np.ndarray):
        contact_map = np.array(contact_map)
    
    indices = np.where(np.triu(contact_map, k=1) == 1)
    pairs = list(zip(indices[0], indices[1]))
    return pairs

def draw_rna_structure(sequence, pairs, save_path, bpseq_path):
	if not pairs:        
		print(f"Skipping structure for {os.path.basename(save_path)} as it has no base pairs.")
		return

	# dot_bracket_string = pairs_to_dot_bracket(pairs, len(sequence))
	# bg = fgb.BulgeGraph.from_dotbracket(dot_bracket_string)
 
	bg = forgi.load_rna(bpseq_path, allow_many=False)
 
	fig, ax = plt.subplots(figsize=(12, 12))
	fvm.plot_rna(bg, ax=ax, text_kwargs={"fontweight": "black"}, lighten=0.7, backbone_kwargs={'linewidth': 3})
	# fvm.plot_rna(cg, text_kwargs={"fontweight":"black"}, lighten=0.7, backbone_kwargs={"linewidth":3})
	fig.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.close(fig)

def save_results(contact_maps, data_generator, output_dir):
	bpseq_dir = os.path.join(output_dir, 'bpseq')
	graph_dir = os.path.join(output_dir, 'structure_graph')
	
	os.makedirs(bpseq_dir, exist_ok=True)
	os.makedirs(graph_dir, exist_ok=True)

	print(f"Saving results to {output_dir}...")
	
	for i, contact_map in enumerate(tqdm(contact_maps)):
		name = data_generator.name[i]
		sequence = data_generator.seqs[i]
		
		filename = "".join(c for c in name if c.isalnum() or c in ('-', '_')).rstrip()
		bpseq_content = contact_map_to_bpseq(sequence, contact_map)
		bpseq_path = os.path.join(bpseq_dir, f"{filename}.bpseq")
		with open(bpseq_path, 'w') as f:
			f.write(bpseq_content)
			
		base_pairs = contact_map_to_pairs(contact_map)

		graph_path = os.path.join(graph_dir, f"{filename}_graph.png")
		draw_rna_structure(sequence, base_pairs, graph_path, bpseq_path)