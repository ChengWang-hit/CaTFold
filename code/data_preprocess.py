import collections

import _pickle as cPickle
from tqdm import tqdm
from utils import seq2pairs, encoding2seq
import random
import numpy as np
import os
from multiprocessing import Pool

class DataGenerator(object):
    def __init__(self, data_dir, split, node_input_dim=128):
        self.data_dir = data_dir
        self.split = split
        self.node_input_dim = node_input_dim
        self.load_data()

    def load_data(self):
        data_dir = self.data_dir
        # Load the current split
        with open(os.path.join(data_dir, '%s.pickle' % self.split), 'rb') as f:
            self.data = cPickle.load(f)
        
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

def max_length(data, max_len=600):
    del_num = 0
    data_max_list = []
    for d in tqdm(data.data):
        if d.length <= 600:
            data_max_list.append(d)

        if d.length > max_len:
            del_num += 1
        
    print(f'Delete num: {del_num}')
    
    with open(f'{data.data_dir}/{data.split}_max600.pickle', 'wb') as f:
        cPickle.dump(data_max_list, f)

# calculate legal pairs (constraint (i))
def func_legal_pairs(data, file_dir):
    if not os.path.exists(f'{data.data_dir}/{file_dir}'):
        os.mkdir(f'{data.data_dir}/{file_dir}')

    idx_list = list(range(data.len))
    random.shuffle(idx_list)
    for i in tqdm(idx_list):
        if os.path.exists(f'{data.data_dir}/{file_dir}/idx_{i}.pickle'):
            continue

        seq = data.seqs[i]
        legal_pairs = np.array(seq2pairs(seq))

        with open(f'{data.data_dir}/{file_dir}/idx_{i}.pickle', 'wb') as f:
            cPickle.dump(legal_pairs, f)

def get_index(data, file_dir):
    index_list = []
    for i in range(data.len):
        # if data.seq_length[i] <= 600:
        #     index_list.append(i)
        index_list.append(i)
    
    with open(f'{data.data_dir}/{file_dir}/index_map_{data.split}.pickle', 'wb') as f:
        cPickle.dump(np.array(index_list), f)

if __name__=="__main__":
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

    # retain data of length up to 600
    max_length(DataGenerator('data/ArchiveII', 'all'))
    os.rename('data/ArchiveII/all_max600.pickle', 'data/ArchiveII/max600.pickle')
    max_length(DataGenerator('data/RNAStralign', 'train_filtered'))
    max_length(DataGenerator('data/RNAStralign', 'test'))

    # Calculating mask matrix (Using multiple processes is highly recommended)
    func_legal_pairs(DataGenerator('data/ArchiveII', 'all'), 'legal_pairs_all')
    get_index(DataGenerator('data/ArchiveII', 'all'), 'legal_pairs_all')

    func_legal_pairs(DataGenerator('data/ArchiveII', 'max600'), 'legal_pairs_max600')
    get_index(DataGenerator('data/ArchiveII', 'max600'), 'legal_pairs_max600')

    func_legal_pairs(DataGenerator('data/RNAStralign', 'train_filtered'), 'legal_pairs_train_filtered')
    get_index(DataGenerator('data/RNAStralign', 'train_filtered'), 'legal_pairs_train_filtered')

    func_legal_pairs(DataGenerator('data/RNAStralign', 'train_filtered_max600'), 'legal_pairs_train_filtered_max600')
    get_index(DataGenerator('data/RNAStralign', 'train_filtered_max600'), 'legal_pairs_train_filtered_max600')

    func_legal_pairs(DataGenerator('data/RNAStralign', 'test'), 'legal_pairs_test')
    get_index(DataGenerator('data/RNAStralign', 'test'), 'legal_pairs_test')

    func_legal_pairs(DataGenerator('data/RNAStralign', 'test_max600'), 'legal_pairs_test_max600')
    get_index(DataGenerator('data/RNAStralign', 'test_max600'), 'legal_pairs_test_max600')