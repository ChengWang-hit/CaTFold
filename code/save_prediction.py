import pickle
import _pickle as cPickle
import os
import collections
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from data_generator import DataGenerator, Dataset
from network import TaGFold
from utils import *

def load_data(dataset, split, results):
    with open(os.path.join('data', dataset, '%s.pickle' % split), 'rb') as f:
        data = cPickle.load(f)
    
    for i in tqdm(range(len(data))):
        results['base_pairs_true'].append(data[i].pairs)

        results['length'].append(data[i].length)
        results['base_pairing_ratio'].append(len(data[i].pairs) / data[i].length * 100)
        try:
            type_ = data[i].name.split('/')[2][:-9]
        except:
            type_ = data[i].name.split('_')[0]
        results['family'].append(type_)

        results['name'].append(data[i].name)

def f1_basepairs(dataset, split, results):
    params = {'batch_size': 10,
              'shuffle': False,
              'num_workers': 5,
              'drop_last': False,
              'collate_fn': collate}
    
    test_data = DataGenerator(f'data/{dataset}', split)
    test_dataset = Dataset(test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    # load model
    model = TaGFold(ct_layer_num=6)
    model.cuda()
    model.load_state_dict(torch.load('checkpoint/CaTFold_best.pt', map_location="cuda" if torch.cuda.is_available() else "cpu")['model'])
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):

            node_onehot, trans_pe, constrained_index, constrained_index_length, label, seq_length, label_index, node_set1, att_mask = data
            node_onehot = node_onehot.cuda()
            trans_pe = trans_pe.cuda()
            label = label.cuda()
            label_index = label_index.cuda()
            att_mask = att_mask.cuda()

            input = (node_onehot, trans_pe, constrained_index, constrained_index_length, seq_length, label_index, att_mask)
            pred = model(input)

            cum = 0
            for j in range(len(seq_length)):

                l = label[cum:cum + seq_length[j] * seq_length[j]].reshape(seq_length[j], seq_length[j])
                p = pred[cum:cum + seq_length[j] * seq_length[j]].reshape(seq_length[j], seq_length[j])
                p = torch.sigmoid(p)

                param = (p, node_set1[j], 0.5)
                p_p = post_process_HK(param)

                results['base_pairs_pred'].append(torch.nonzero(p_p).tolist())

                results['f1'].append(evaluate(p_p, l)[-1].item())
                cum += seq_length[j] * seq_length[j]

'''
Calculate and save the statistics for plotting
'''
if __name__=='__main__':
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    dataset, split = 'ArchiveII', 'all'
    results = {
               'f1':[],
               'length':[],
               'base_pairing_ratio':[],
               'pseudoknot':[],
               'family':[],
               'base_pairs_pred':[],
               'base_pairs_true':[],
               'name':[]
              }
    load_data(dataset, split, results)
    f1_basepairs(dataset, split, results)

    with open(f'results/{dataset}_{split}.pkl', 'wb') as f:
        pickle.dump(results, f)

    dataset, split = 'RNAStralign', 'test'
    results = {
               'f1':[],
               'length':[],
               'base_pairing_ratio':[],
               'pseudoknot':[],
               'family':[],
               'base_pairs_pred':[],
               'base_pairs_true':[],
               'name':[]
              }
    load_data(dataset, split, results)
    f1_basepairs(dataset, split, results)

    with open(f'results/{dataset}_{split}.pkl', 'wb') as f:
        pickle.dump(results, f)