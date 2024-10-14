import collections

import _pickle as cPickle
from tqdm import tqdm
from data_generator import DataGenerator
from utils import seq2pairs
import random
import numpy as np
from sys import getsizeof

import os
os.chdir('/home/wangcheng/project/RNA/TaGFoldv3')
import torch

def cut_length(data, max_len=600):
    cut_num = 0
    data_cut_list = []
    for d in tqdm(data.data):
        one_hot_matrix_cut = d.seq[:max_len]
        ss_label_cut = d.seq[:max_len]
        seq_len = d.length if d.length < max_len else max_len
        seq_name = d.name

        pairs_cut = []
        for pair in d.pairs:
            if pair[0] < max_len and pair[1] < max_len:
                pairs_cut.append(pair)

        data_cut = RNA_SS_data(seq=one_hot_matrix_cut, ss_label=ss_label_cut, length=seq_len, name=seq_name, pairs=pairs_cut)
        data_cut_list.append(data_cut)

        if d.length > max_len:
            cut_num += 1
        
    print(f'Cut num: {cut_num}')
    
    with open(f'{data.data_dir}/{split}_cuta600.pickle', 'wb') as f:
        cPickle.dump(data_cut_list, f)

def max_length(data, max_len=600):
    del_num = 0
    data_max_list = []
    for d in tqdm(data.data):
        if d.length <= 600:
            data_max_list.append(d)

        if d.length > max_len:
            del_num += 1
        
    print(f'Delete num: {del_num}')
    
    with open(f'{data.data_dir}/{split}_max600.pickle', 'wb') as f:
        cPickle.dump(data_max_list, f)

# 计算legal pairs并保存
def func_legal_pairs(data):
    if not os.path.exists(f'{data.data_dir}/{file_dir}'):
        os.mkdir(f'{data.data_dir}/{file_dir}')

    idx_list = list(range(data.len))
    random.shuffle(idx_list)
    for i in tqdm(idx_list):
        if os.path.exists(f'{data.data_dir}/{file_dir}/idx_{i}.pickle'):
            continue
        # if os.path.getsize(f'{data.data_dir}/legal_pairs/idx_{i}.pickle') > 0:
        #     continue
        seq = data.seqs[i]
        legal_pairs = np.array(seq2pairs(seq))
        # print(f'list {round(getsizeof(legal_pairs) / 1024, 2)} KB')
        # print(f'array {round(getsizeof(np.array(legal_pairs, dtype=int)) / 1024, 2)} KB')
        # print(f'tensor {round(getsizeof(torch.LongTensor(legal_pairs)) / 1024, 2)} KB')

        with open(f'{data.data_dir}/{file_dir}/idx_{i}.pickle', 'wb') as f:
            cPickle.dump(legal_pairs, f)

def get_index(data):
    index_list = []
    for i in range(data.len):
        # if data.seq_length[i] <= 600:
        #     index_list.append(i)
        index_list.append(i)
    
    with open(f'{data.data_dir}/{file_dir}/index_map_{data.split}.pickle', 'wb') as f:
        cPickle.dump(np.array(index_list), f)

if __name__=="__main__":
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    # dataset = 'ArchiveII'
    # split = 'max600'
    
    dataset = 'RNAStralign'
    split = 'test_max600'

    global file_dir
    file_dir = f'legal_pairs_{split}'

    train_data_all = DataGenerator(f'data/{dataset}', f'{split}')
    # max_length(train_data_all)
    func_legal_pairs(train_data_all)
    get_index(train_data_all)