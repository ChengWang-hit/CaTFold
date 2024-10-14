import collections
import os
from multiprocessing import Pool

import _pickle as cPickle
import numpy as np
from torch.utils import data
from tqdm import tqdm
from utils import encoding2seq
import torch

class DataGenerator(object):
    '''
    construct data: 
    1 transformer input
        1.1 one-hot embedding
        1.2 position embedding
    2 graph input
        2.1 node
        2.2 edge
    '''
    def __init__(self, data_dir, split, node_input_dim=128):
        self.data_dir = data_dir
        self.split = split
        self.node_input_dim = node_input_dim
        self.load_data()

    def load_data(self):
        data_dir = self.data_dir
        # Load the current split
        RNA_SS_data = collections.namedtuple('RNA_SS_data', 
            'seq ss_label length name pairs')
        with open(os.path.join(data_dir, '%s.pickle' % self.split), 'rb') as f:
            self.data = cPickle.load(f)
        
        with open(f'{self.data_dir}/legal_pairs_{self.split}/index_map_{self.split}.pickle', 'rb') as f:
            self.index_map = cPickle.load(f)
        
        self.node_onehot, self.contact_pairs, self.contact_maps, self.chain_pairs, self.seq_length = [], [], [], [], []
        for i in tqdm(range(len(self.data))):
            # no need padding
            self.node_onehot.append(self.data[i].seq[:self.data[i].length])

            self.contact_pairs.append(self.data[i].pairs)
            
            self.seq_length.append(self.data[i].length)

        self.len = len(self.node_onehot) # data num
        p = Pool()
        self.seqs = list(p.map(encoding2seq, self.node_onehot)) # RNA sequence
        p.close()

        self.node_position = []
        for i, seq in enumerate(self.seqs):
            # self.node_position.append(relative_position(seq))
            self.node_position.append(torch.arange(0, self.seq_length[i]))

    def pairs2map(self, pairs, seq_len):
        contact = torch.zeros([seq_len, seq_len])
        idx = torch.LongTensor(pairs).T
        contact[idx[0], idx[1]] = 1
        return contact

    # node set for bipartite graph: {A,G}, {C,U}
    def seq2set(self, seq):
        set1 = {'A', 'G'}
        node_set1 = []
        for i, s in enumerate(seq):
            if s in set1:
                node_set1.append(i)
        return node_set1

    def get_one_sample(self, index):
        '''
        input feature
        '''
        feature = self.node_onehot[index]
        seq_length = self.seq_length[index]
        seq = self.seqs[index]
        contact_pairs = self.contact_pairs[index]
        if not contact_pairs:
            contact_pairs = [[0, 0]]

        # label
        contact_map = self.pairs2map(contact_pairs, seq_length) # torch操作

        # position embedding
        trans_pe = torch.zeros(seq_length, self.node_input_dim, dtype=torch.float)
        position = self.node_position[index].unsqueeze(dim=1).float()
        div_term = (10000 ** ((2*torch.arange(0, self.node_input_dim/2)) / self.node_input_dim)).unsqueeze(dim=1).T
        trans_pe[:, 0::2] = torch.sin(position @ (1/div_term))  # 偶数维度
        trans_pe[:, 1::2] = torch.cos(position @ (1/div_term))  # 奇数维度

        with open(f'{self.data_dir}/legal_pairs_{self.split}/idx_{int(self.index_map[index])}.pickle', 'rb') as f:
            legal_pairs = cPickle.load(f)

        constrained_index = torch.LongTensor(legal_pairs).transpose(1, 0)
        constrained_index_length = constrained_index.shape[1]

        edges = constrained_index.transpose(1, 0)
        att_mask = torch.ones(seq_length, seq_length, dtype=torch.bool)
        att_mask[edges[:, 0], edges[:, 1]] = False
        att_mask.fill_diagonal_(False)

        # bipartite graph node set
        node_set1 = self.seq2set(seq)

        if len(contact_pairs) == 0:
            contact_pairs = [[0, 0]]
        label_index = torch.LongTensor(contact_pairs).transpose(1, 0)

        return torch.Tensor(feature), trans_pe, \
               constrained_index, constrained_index_length, \
               torch.Tensor(contact_map).view(-1), seq_length, \
               label_index, node_set1, \
               att_mask

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data.seq_length)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data.get_one_sample(index)