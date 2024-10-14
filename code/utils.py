import argparse
import os
import random
from torch.nn.utils.rnn import pad_sequence
import networkx as nx
from scipy import signal

import numpy as np
import torch
import sys
import pickle

class Logger(object):
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(fileN, 'w')

    def write(self, message):
        '''sys.stdout.write'''
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

property_vec_dict = {
    'A':np.array([1,1,1]),
    'U':np.array([0,0,1]),
    'G':np.array([1,0,0]),
    'C':np.array([0,1,0])
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
        default='/code/config.json',
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
    # retain the maximum value by row
    max_values, max_indices = torch.max(p, dim=1)
    p = torch.where(p == max_values.view(-1, 1), p, torch.zeros_like(p))
    
    # retain the values greater than threshold
    p = torch.where(p > threshold, torch.ones_like(p), torch.zeros_like(p))
    return p.cuda()

def post_process_blossom(mat, threshold=0.5):
    # retain the values greater than threshold
    mat = torch.where(mat > threshold, mat, torch.zeros_like(mat)).detach().cpu()
    n = mat.size(-1)
    G = nx.convert_matrix.from_numpy_array(np.array(mat.data))
    pairings = nx.matching.max_weight_matching(G)
    y_out = torch.zeros_like(mat)
    for (i, j) in pairings:
        if i>n and j>n:
            continue
        y_out[i%n, j%n] = 1
        y_out[j%n, i%n] = 1
    return y_out.cuda()

# hopcroft_karp
def post_process_HK(param):
    mat, node_set1, threshold = param

    mat = torch.where(mat > threshold, mat, torch.zeros_like(mat))
    n = mat.size(-1)
    G = nx.convert_matrix.from_numpy_array(np.array(mat.data.cpu()))
    pairings = nx.bipartite.maximum_matching(G, top_nodes=node_set1)
    y_out = torch.zeros_like(mat)
    for (i, j) in pairings.items():
        if i>n and j>n:
            continue
        y_out[i%n, j%n] = 1
        y_out[j%n, i%n] = 1
    return y_out.cuda()

# wether satisfying constraint (i)
def check_illegal(seq, true_pairs):
    legal_base_pair = set({'AU', 'UA', 'GC', 'CG', 'GU', 'UG'})
    for pair in true_pairs:
        c, r = pair
        if seq[c]+seq[r] not in legal_base_pair:
            return True
        
        else:
            if abs(c-r) < 4:
                return True

    return False

# one-hot to sequence
def encoding2seq(arr):
    seq = list()
    for arr_row in list(arr):
        # if sum(arr_row)==0:
        #     seq.append('.')
        # else:
        #     seq.append(char_dict[np.argmax(arr_row)])

        seq.append(char_dict[np.argmax(arr_row)])
    return ''.join(seq)

def seq2feature(params):
    onehot, seq = params
    feature = property_vec_dict[seq[0]]
    for i in range(1, len(seq)):
        feature = np.vstack((feature, property_vec_dict[seq[i]]))

    return np.hstack((onehot, feature))

def load_legal_pairs(filename):
    file_size = os.path.getsize(filename)/1024
    if file_size > 1000:
        with open(filename, 'rb') as f:
            legal_pairs = pickle.load(f)
        return legal_pairs
    else:
        return None

def seq2pairs(s):
    pairs_symmetry = []

    for i in range(len(s) - 4):
        for j in range(i+4, len(s)):
            if (s[i] == 'A' and s[j] == 'U') or (s[i] == 'U' and s[j] == 'A'):
                pairs_symmetry.append([i, j])
                pairs_symmetry.append([j, i])
            
            if (s[i] == 'G' and s[j] == 'C') or (s[i] == 'C' and s[j] == 'G'):
                pairs_symmetry.append([i, j])
                pairs_symmetry.append([j, i])
            
            if (s[i] == 'G' and s[j] == 'U') or (s[i] == 'U' and s[j] == 'G'):
                pairs_symmetry.append([i, j])
                pairs_symmetry.append([j, i])

    return pairs_symmetry

def collate(data_list):
    one_hot_list = []
    trans_pe_list = []
    att_mask_pad_list = []
    constrained_index_list = []
    constrained_index_length_list = []
    labels_list = []
    seq_length_list = [data[5] for data in data_list]
    label_index_list = []
    node_set1_list = []
    max_length = max(seq_length_list)

    offset = 0
    for i in range(len(data_list)):
        one_hot_list.append(data_list[i][0])
        
        trans_pe_list.append(data_list[i][1])

        att_mask = data_list[i][8]
        att_mask_pad = torch.zeros(max_length, max_length, dtype=torch.bool)
        att_mask_pad[:att_mask.size(0), :att_mask.size(1)] = att_mask
        att_mask_pad[:, att_mask.size(0):] = True
        att_mask_pad_list.append(att_mask_pad)

        constrained_index_list.append(data_list[i][2]+offset) # with offset

        constrained_index_length_list.append(data_list[i][3])

        labels_list.append(data_list[i][4])

        label_index_list.append(data_list[i][6])

        node_set1_list.append(data_list[i][7])

        offset += seq_length_list[i]

    return pad_sequence(one_hot_list, batch_first=True), \
           pad_sequence(trans_pe_list, batch_first=True), \
           torch.cat(constrained_index_list, dim=1), \
           torch.LongTensor(constrained_index_length_list), \
           torch.cat(labels_list, dim=0), \
           torch.LongTensor(seq_length_list), \
           torch.cat(label_index_list, dim=1), \
           node_set1_list, \
           torch.stack(att_mask_pad_list)
