'''
对数据的分布进行统计分析
'''
import collections
import pickle
import _pickle as cPickle
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from itertools import groupby
import numpy as np
from tqdm import tqdm
from utils import seq2pairs
import seaborn as sns
import matplotlib.pyplot as plt
from data_generator import DataGenerator, Dataset
from network import TaGFold
import pandas as pd
import torch
from utils import *
import matplotlib.ticker as mtick

# sns.set(style='whitegrid')
sns.set_style("white")
# sns.despine()
# sns.despine(offset=10)

from matplotlib import rcParams
rcParams['figure.figsize'] = 6.4, 3.8

def load_data(dataset, split):
    with open(os.path.join('data', dataset, '%s.pickle' % split), 'rb') as f:
        data = cPickle.load(f)
    return data

def seq_len_distribution(data):
    interval = 100
    length_list = [data[i].length for i in range(len(data))]
    length_dist = groupby(sorted(length_list), key=lambda x: x//interval)
    # print(length_dist)
    for k, g in length_dist:
        print('{}-{}: {}'.format(k*interval, (k+1)*interval-1, len(list(g))))

def length_distribution(data_train, data_val=None, data_test=None, dataset=None):
    # length_list = [data_train[i].length for i in range(len(data_train))] + \
    #               [data_val[i].length for i in range(len(data_val))] + \
    #               [data_test[i].length for i in range(len(data_test))]
    length_list = [data_train[i].length for i in range(len(data_train))]
    data_len = pd.DataFrame({'length':length_list})
    print('*************length distribution************')
    print(f'min: {min(length_list)}')
    print(f'max: {max(length_list)}')

    sns.histplot(data=data_len, x='length', kde=True, bins=50, stat="probability")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.17)
    plt.ylabel('Probability Density')
    plt.xlabel('Length of RNA')
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.savefig(f'results/{dataset}_length.png', dpi=600)
    plt.cla()

def family_distribution(data):
    family_len = {}
    for d in data:
        try:
            type_ = d.name.split('/')[2]
        except:
            type_ = d.name.split('_')[0]
        if type_ not in family_len.keys():
            family_len[type_] = [d.length]
        else:
            family_len[type_].append(d.length)

    for key in family_len.keys():
        print(f'type: {key}')
        print(f'num: {len(family_len[key])}')
        print(f'min: {min(family_len[key])}\tmax: {max(family_len[key])}')
        print()
    return

'''
Compare the speed of Blossom and HK(ArchiveII)
'''
def Blossom_vs_HK(dataset, split):
    Blossom_list, HK_list = [], []
    length_list = []

    if not os.path.exists('results/Blossom_vs_HK.pickle'):
        params = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 1,
                'drop_last': False,
                'collate_fn': collate}
        test_data = DataGenerator(f'data/{dataset}', split)
        test_dataset = Dataset(test_data)
        test_loader = torch.utils.data.DataLoader(test_dataset, **params)

        model = TaGFold(ct_layer_num=6)
        model.cuda()
        model.load_state_dict(torch.load('checkpoint/CaTFold_best.pt', map_location="cuda" if torch.cuda.is_available() else "cpu")['model'])
        model.eval()
        result = []
        result_shifted = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):

                # if i > 20:
                #     break

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

                    length_list.append(seq_length[j].item())
                    param = (p, node_set1[j], 0.5)
                    start = time.time()
                    p_p = post_process_HK(param)
                    HK_list.append(time.time()-start)

                    start = time.time()
                    p_p = post_process_blossom(p)
                    Blossom_list.append(time.time()-start)

                    # print(evaluate(p_p, l)[-1].item())

                    # result.append(r)
                    # result_shifted.append(evaluate_shifted(p_p, l))
                    cum += seq_length[j] * seq_length[j]

        dict_data = {'Length':length_list+length_list, 'Time':Blossom_list+HK_list, 'Algorithm':['Blossom']*len(Blossom_list) + ['Hopcroft_Karp']*len(HK_list)}

        with open('results/Blossom_vs_HK.pickle', 'wb') as f:
            cPickle.dump(dict_data, f)
    else:
        dict_data = cPickle.load(open(f'results/Blossom_vs_HK.pickle',"rb"))

    plot_data = pd.DataFrame(dict_data)

    sns.lmplot(x='Length', y='Time', hue='Algorithm', data=plot_data, order=2, scatter_kws={"s":10, "alpha":0.2}, height=4.2, aspect=1.43)
    # sns.lmplot(x='Length', y='Blossom', data=plot_data, order=2, scatter_kws={"s": 5})
    # sns.pairplot(plot_data, x_vars=['Length'], y_vars=['Time'], hue='Algorithm', height=5, aspect=.8, kind='reg')
    plt.xlabel('Length of RNA', labelpad=1)
    # plt.gcf().subplots_adjust(bottom=0.38)
    plt.ylabel('Infer time (s)')
    # plt.legend(labels = ['True','Predicted'], loc='upper right')
    # plt.title(f'{dataset}', y=0.97)
    plt.savefig(f'results/{dataset}_infer_time.png', dpi=600)
    plt.cla()

    # sns.lmplot(x='Length', y='HK', data=plot_data, order=2, plot_kws={"s": 5})
    # plt.xlabel('Length of RNA chain', labelpad=1)
    # plt.gcf().subplots_adjust(bottom=0.38)
    # plt.ylabel('Infer time (s)')
    # # plt.legend(labels = ['True','Predicted'], loc='upper right')
    # # plt.title(f'{dataset}', y=0.97)
    # plt.savefig(f'/home/wangcheng/project/RNA/TaGFoldv3/pictures/{dataset}_infer_time.png')
    # plt.cla()

    # p, r, f1 = zip(*result)
    # p_s, r_s, f1_s = zip(*result_shifted)

    # print('precision: ', np.average(p))
    # print('recall: ', np.average(r))
    # print('F1: ', np.average(f1))
    # print()
    # print('precision(S): ', np.average(p_s))
    # print('recall(S): ', np.average(r_s))
    # print('F1(S): ', np.average(f1_s))
    # print()

def infer_time(dataset):
    CaTFold_time = cPickle.load(open(f'results/CaTFold_infertime_ArchiveII600.pickle',"rb"))
    UFold_time = cPickle.load(open(f'results/UFold_infertime_ArchiveII600.pickle',"rb"))
    GCNfold_time = cPickle.load(open(f'results/GCNfold_infertime_ArchiveII600.pickle',"rb"))

    points = {'length':[],
              'time':[],
              'Model':[]}
    
    # print(sum([d[1] for d in CaTFold_time]))
    # print(sum([d[1] for d in UFold_time]))
    # print(sum([d[1] for d in GCNfold_time]))
    
    for (length, time) in CaTFold_time:
        points['length'].append(length)
        points['time'].append(time)
        points['Model'].append('CaTFold')
    
    for (length, time) in UFold_time:
        points['length'].append(length)
        points['time'].append(time)
        points['Model'].append('UFold')

    for (length, time) in GCNfold_time:
        points['length'].append(length)
        points['time'].append(time)
        points['Model'].append('GCNfold')
    
    plot_data = pd.DataFrame(points)
    sns.lmplot(x='length', y='time', hue='Model', order=1, data=plot_data, scatter_kws={"s":5, "alpha":0.2}, height=4.2, aspect=1.43)
    plt.xlabel('Length of RNA', labelpad=1)
    plt.ylabel('Infer time (s)')
    plt.ylim(top=0.12, bottom=0)
    plt.savefig(f'results/{dataset}_infer_time_CaTFold_UFold_GCNfold.png', dpi=600)
    plt.cla()

def RNA_fragment_distribution(data):
    def pairs2map(pairs, seq_len):
        contact = torch.zeros([seq_len, seq_len])
        idx = torch.LongTensor(pairs).T
        contact[idx[0], idx[1]] = 1
        return contact
    
    def frag_len(contact, position, seq_len):
        max_len = 1
        # right up
        x, y = position
        for j in range(y+1, seq_len):
            x -= 1
            if x < 0 or contact[x][j] == 0:
                break
            max_len += 1
            visited.add((x, j))

        x, y = position
        for i in range(x+1, seq_len):
            y -= 1
            if y < 0 or contact[i][y] == 0:
                break
            max_len += 1
            visited.add((i, y))
        return max_len
    
    visited = set()
    len_dist = []

    if os.path.exists(f'results/{dataset}_frag_length.pkl'):
        len_dist = pickle.load(open(f'results/{dataset}_frag_length.pkl',"rb"))
    else:
        for k in tqdm(range(len(data))):
            seq_len, pairs = data[k].length, data[k].pairs
            contact = pairs2map(pairs, seq_len)
            for i in range(seq_len-1):
                for j in range(i+1, seq_len):
                    if contact[i][j] == 1 and (i, j) not in visited:
                        length = frag_len(contact, (i, j), seq_len)
                        len_dist.append(length)

        pickle.dump(len_dist, open(f'results/{dataset}_frag_length.pkl',"wb"))

    print('*************length distribution************')
    print(f'min: {min(len_dist)}')
    print(f'max: {max(len_dist)}')

    # plt.figure(figsize=(12, 12))
    # plt.imshow(contact)
    # plt.savefig(os.path.join(f'/home/wangcheng/project/RNA/TaGFoldv3/pictures/{dataset}_contactmap.png'))
    # plt.cla()

    sns.histplot(data=len_dist, bins=50, stat="probability")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.17)
    plt.ylabel('Probability Density')
    plt.xlabel('Length of RNA fragment')
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.savefig(f'results/{dataset}_frag_length.png', dpi=600)
    plt.cla()
    
    # pickle.dump(len_dist, open(f'/home/wangcheng/project/RNA/TaGFoldv3/pictures/{dataset}_frag_length.pkl',"wb"))
    # len_dist_saved = pickle.load(open(f'/home/wangcheng/project/RNA/TaGFoldv3/pictures/{dataset}_frag_length.pkl',"rb"))

    print()
    # print(len_dist)

'''
F1 vs length
'''
def F1_length(dataset, split):
    # 5S, 16S, tRNA, SRP, Group I Intron, RNaseP, tmRNA, telomerase. 23S and Group II Intron
    idx_rank = [[],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                []]
    family_map = {'5S_rRNA':'5S', '5s':'5S',
                  '16S_rRNA':'16S', '16s':'16S',
                  'tRNA':'tRNA',
                  'SRP':'SRP', 'srp':'SRP',
                  'group_I_intron':'Group I Intron', 'grp1':'Group I Intron',
                  'RNaseP':'RNaseP',
                  'tmRNA':'tmRNA',
                  'telomerase':'telomerase',
                  '23s':'23S',
                  'grp2':'Group II Intron'}
    
    with open(f'results/{dataset}_{split}.pkl', 'rb') as f:
        results = cPickle.load(f)

    for i, family in enumerate(results['family']):
        if '5S_rRNA' == family or '5s' == family:
            idx_rank[0].append(i)
        elif '16S_rRNA' == family or '16s' == family:
            idx_rank[1].append(i)
        elif 'tRNA' == family:
            idx_rank[2].append(i)
        elif 'SRP' == family or 'srp' == family:
            idx_rank[3].append(i)
        elif 'group_I_intron' == family or 'grp1' == family:
            idx_rank[4].append(i)
        elif 'RNaseP' == family:
            idx_rank[5].append(i)
        elif 'tmRNA' == family:
            idx_rank[6].append(i)
        elif 'telomerase' == family:
            idx_rank[7].append(i)
        elif '23s' == family:
            idx_rank[8].append(i)
        elif 'grp2' == family:
            idx_rank[9].append(i)

    f1_list = []
    length_list = []
    family_list = []

    for idx in sum(idx_rank,[]):
        f1_list.append(results['f1'][idx])
        length_list.append(results['length'][idx])
        family_list.append(family_map[results['family'][idx]])
    
    plot_data = pd.DataFrame({'f1':f1_list, 
                              'length':length_list,
                              'family':family_list})

    sns.relplot(data=plot_data, x="length", y="f1", hue="family", s=12, alpha=0.5, height=4.2, aspect=1.43)
    plt.xlabel('Length of RNA', labelpad=1)
    plt.ylabel('F1-Score')

    # RNAStralign
    if dataset == 'RNAStralign':
        plt.xlim(right=1600)
        plt.ylim(bottom=0.5)

    # ArchiveII
    if dataset == 'ArchiveII':
        plt.xlim(right=1000)
        plt.ylim(bottom=0.2)

    plt.savefig(f'results/{dataset}_{split}_f1_length.png', dpi=600)
    plt.cla()

'''
F1 vs base pair
'''
def F1_ratio(dataset, split):
    with open(f'results/{dataset}_{split}.pkl', 'rb') as f:
        results = cPickle.load(f)
    
    idx_rank = [[],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                []]
    family_map = {'5S_rRNA':'5S', '5s':'5S',
                  '16S_rRNA':'16S', '16s':'16S',
                  'tRNA':'tRNA',
                  'SRP':'SRP', 'srp':'SRP',
                  'group_I_intron':'Group I Intron', 'grp1':'Group I Intron',
                  'RNaseP':'RNaseP',
                  'tmRNA':'tmRNA',
                  'telomerase':'telomerase',
                  '23s':'23S',
                  'grp2':'Group II Intron'}

    for i, family in enumerate(results['family']):
        if '5S_rRNA' == family or '5s' == family:
            idx_rank[0].append(i)
        elif '16S_rRNA' == family or '16s' == family:
            idx_rank[1].append(i)
        elif 'tRNA' == family:
            idx_rank[2].append(i)
        elif 'SRP' == family or 'srp' == family:
            idx_rank[3].append(i)
        elif 'group_I_intron' == family or 'grp1' == family:
            idx_rank[4].append(i)
        elif 'RNaseP' == family:
            idx_rank[5].append(i)
        elif 'tmRNA' == family:
            idx_rank[6].append(i)
        elif 'telomerase' == family:
            idx_rank[7].append(i)
        elif '23s' == family:
            idx_rank[8].append(i)
        elif 'grp2' == family:
            idx_rank[9].append(i)

    f1_list = []
    base_pairing_ratio_list = []
    family_list = []

    for idx in sum(idx_rank,[]):
        f1_list.append(results['f1'][idx])
        base_pairing_ratio_list.append(results['base_pairing_ratio'][idx])
        family_list.append(family_map[results['family'][idx]])

    plot_data = pd.DataFrame({'f1':f1_list, 
                              'base_pairing_ratio':base_pairing_ratio_list,
                              'family':family_list})
    
    sns.relplot(data=plot_data, x="base_pairing_ratio", y="f1", hue="family", s=12, alpha=0.5, height=4.2, aspect=1.43)
    plt.xlabel('Base-pairing ratio(%)', labelpad=1)
    plt.ylabel('F1-Score')

    # RNAStralign
    if dataset == 'RNAStralign':
        # plt.xlim(right=1600)
        plt.ylim(bottom=0.5)

    # ArchiveII
    if dataset == 'ArchiveII':
        # plt.xlim(right=1000)
        plt.ylim(bottom=0.2)

    plt.savefig(f'results/{dataset}_{split}_f1_ratio.png', dpi=600)
    plt.cla()

if __name__=='__main__':
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    dataset, split = 'RNAStralign', 'train_filtered'
    data = load_data(dataset, split)
    RNA_fragment_distribution(data)
    seq_len_distribution(data)
    length_distribution(data, None, None, dataset)
    family_distribution(data)

    dataset, split = 'RNAStralign', 'test'
    data = load_data(dataset, split)
    F1_length(dataset, split)
    F1_ratio(dataset, split)

    dataset, split = 'ArchiveII', 'all'
    data_all = load_data(dataset, split)
    length_distribution(data_all, None, None, dataset)
    family_distribution(data_all)
    F1_length(dataset, split)
    F1_ratio(dataset, split)

    Blossom_vs_HK('ArchiveII', 'all')
    # infer_time('ArchiveII')