import os

import _pickle as cPickle
from torch.utils import data
from utils import *
import torch
from multiprocessing import Pool

class DataGenerator(object):
    def __init__(self, data_dir, split, node_input_dim, mode='test', family_fold=False, data_num=999999):
        self.data_dir = data_dir
        self.split = split
        self.node_input_dim = node_input_dim
        self.mode = mode
        if family_fold:
            self.mask_dir = self.data_dir.replace('/family_fold', '')
        else:
            self.mask_dir = self.data_dir
        self.load_data(data_num)

    def load_data(self, data_num):
        data_dir = self.data_dir

        with open(os.path.join(data_dir, f'{self.split}.pickle'), 'rb') as f:
            self.data = cPickle.load(f)
        
        # test_num = int(100000) # test
        self.seq = [seq.upper() for seq in self.data['seq'][:data_num]]
        self.ss = self.data['ss'][:data_num]
        self.mask_matrix_idx = self.data['mask_matrix_idx'][:data_num]
        self.len = len(self.seq)

    def merge(self, augmentation):
        self.seq = self.seq + augmentation.seq
        self.ss = self.ss + augmentation.ss
        self.mask_matrix_idx = self.mask_matrix_idx + augmentation.mask_matrix_idx
        self.len = len(self.seq)

        return self

    def get_one_sample(self, index):
        seq = self.seq[index]
        node_onehot = seq2onehot(seq)
        seq_length = len(seq)

        # sinusoidal position embedding
        node_pe = absolute_position_embedding(seq_length, self.node_input_dim)

        if self.mode == 'finetune':
    
            # ss
            cm_pairs = np.array(self.ss[index]) # (N, 2)
            if len(cm_pairs) == 0:
                cm_pairs = np.array([[0, seq_length-1], [seq_length-1, 0]])
            contact_map = pairs2map(cm_pairs, seq_length)
            row_indices, col_indices = torch.triu_indices(seq_length, seq_length, offset=1) # Upper triangle
            pred_pairs = torch.vstack((row_indices, col_indices))
            label_ss = contact_map[pred_pairs[0, :], pred_pairs[1, :]]

            result = (
                    torch.Tensor(node_onehot),
                    torch.Tensor(node_pe),															
                    seq_length,
                    torch.Tensor(label_ss))
        else:
            with open(os.path.join(self.mask_dir, 'mask_matrix', f'{self.mask_matrix_idx[index]}.pickle'), 'rb') as f:
                pred_pairs = torch.LongTensor(cPickle.load(f)).T # (2, N)
            mask_matrix = torch.zeros((seq_length, seq_length))
            mask_matrix[pred_pairs[0, :], pred_pairs[1, :]] = 1
            mask_matrix = mask_matrix + mask_matrix.T

            cm_pairs = np.array(self.ss[index])
            if len(cm_pairs) == 0:
                cm_pairs = np.array([[0, seq_length-1], [seq_length-1, 0]])
            contact_map = pairs2map(cm_pairs, seq_length)

            # bipartite graph
            node_set1 = seq2set(seq)

            result = (
                   torch.Tensor(node_onehot), \
                torch.Tensor(node_pe), \
                torch.Tensor(mask_matrix), \
                torch.Tensor(contact_map), \
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