import os

import torch
from utils import *
from config import *
from network import TaGFold

def infering(model, data, threshold=0.5):
    print('testing……')
    model.eval()
    
    with torch.no_grad():
        node_onehot, trans_pe, constrained_index, constrained_index_length, seq_length, att_mask, node_set1 = data
        node_onehot = node_onehot.cuda()
        trans_pe = trans_pe.cuda()
        att_mask = att_mask.cuda()

        input = (node_onehot, trans_pe, constrained_index, constrained_index_length, seq_length, None, att_mask)
        pred = torch.sigmoid(model(input))

        p = pred[:seq_length[0] * seq_length[0]].reshape(seq_length[0], seq_length[0])
        param = (p, node_set1, threshold)
        contact_map = post_process_HK(param)

        base_pairs = torch.nonzero(contact_map)

    return base_pairs.cpu().numpy()

def seq2set(seq):
    set1 = {'A', 'G'}
    node_set1 = []
    for i, s in enumerate(seq):
        if s in set1:
            node_set1.append(i)
    return node_set1

def seq2onehot(seq):
    char_dict = {
        'A':[1,0,0,0],
        'U':[0,1,0,0],
        'C':[0,0,1,0],
        'G':[0,0,0,1]
        }
    
    node_onehot = [char_dict[c] for c in seq]
    return torch.Tensor(node_onehot)

def preprocess_data(input_RNA):
    node_input_dim = 128

    seq_length = len(input_RNA)
    
    node_onehot = seq2onehot(input_RNA).unsqueeze(0)

    # position embedding
    trans_pe = torch.zeros(seq_length, node_input_dim, dtype=torch.float)
    position = torch.arange(0, seq_length).unsqueeze(dim=1).float()
    div_term = (10000 ** ((2*torch.arange(0, node_input_dim/2)) / node_input_dim)).unsqueeze(dim=1).T
    trans_pe[:, 0::2] = torch.sin(position @ (1/div_term))
    trans_pe[:, 1::2] = torch.cos(position @ (1/div_term))
    trans_pe = trans_pe.unsqueeze(0)

    legal_pairs = torch.Tensor(seq2pairs(input_RNA))
    constrained_index = torch.LongTensor(legal_pairs.numpy()).transpose(1, 0)
    constrained_index_length = torch.LongTensor(constrained_index.shape[1])

    edges = constrained_index.transpose(1, 0)
    att_mask = torch.ones(seq_length, seq_length, dtype=torch.bool)
    att_mask[edges[:, 0], edges[:, 1]] = False
    att_mask.fill_diagonal_(False)
    att_mask = att_mask.unsqueeze(0)

    # bipartite graph node set
    node_set1 = seq2set(input_RNA)

    return node_onehot, trans_pe, constrained_index, constrained_index_length, torch.LongTensor([seq_length]), att_mask, node_set1

def base_pairs2dot_bracket(input_RNA, base_pairs):
    result = ['.']*len(input_RNA)
    for bp in base_pairs:
        result[min(bp)] = '('
        result[max(bp)] = ')'
    return ''.join(result)

def main():
    # logger = Logger('results/output.txt')
    data = preprocess_data(input_RNA)

    model = TaGFold()
    checkpoint_path = 'checkpoint/CaTFold_best.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")['model'])
    model.cuda()
    
    print('Infering......')
    base_pairs = infering(model, data)
    secondary_struture = base_pairs2dot_bracket(input_RNA, base_pairs)
    print(input_RNA)
    print(secondary_struture)

'''
Example1: Aspergillus fumigatusspecies, the RNA ID is GSP-41122, as recorded in SRPDB database (278 nucleotides)
Seq : ACUUAAUCUGUCAUGGAUAACCUAGUGGAAGGCCUGCCGCUAAGUCAGUAACCUUGCUGCGGCAUUUGCCAGCGGGAAAGGUGCCCGGUACGAAUCCUGGGGUCGUCGUUGUACUCGUGCGAGUAAUCCACGAUGCUACAAGGCGCUAAGCAAUGGAAGUGAAUCUUGAGGGAAGCAAUUCUGCAGAGACACUUCCACCCUGGGAUGGCGUCGCCGGAGGACACCUACCCGUUACAGGGAAGUUGGCUGUUUGGCUGGACAACCGCAAUCUUCUUUUU
Pred: ..(((.(........).)))..(.((((..(.((.((((.((..(((.(..(((...(((((.).)))).((((((.(.((((.((((((((.(((((((((.((((.(.(((.(((((.((....)))))))...)))).))))(...)...(((((((...((((....(((....)))....)))).))))))))))))))))..))).)))))....))))).))))))..)))..).).)).))..)))).)).)..)))).)..........

Example2: Homo sapiens 5S rRNA URS000002B0D5_9606 (120 nucleotides)
Seq : GUCUACGGCCAUACCACCCUGAACGCGCCCGAUCUCGUCUGAUCUCGGAAGCUAAGCAGGGUCGGGCCUGGUUAGUACUUGGAUGGGAGACCGCCUGGGAAUACCGGGUGCUGUAGGCUU
Pred: (((((((((....((((((((.....((((((............))))..))....)))))).)).((((((.....((.((.(((....))))).))....)))))).)))))))))..

Visulize on http://rna.tbi.univie.ac.at/forna/.
'''
if __name__ == '__main__':
    input_RNA = 'ACUUAAUCUGUCAUGGAUAACCUAGUGGAAGGCCUGCCGCUAAGUCAGUAACCUUGCUGCGGCAUUUGCCAGCGGGAAAGGUGCCCGGUACGAAUCCUGGGGUCGUCGUUGUACUCGUGCGAGUAAUCCACGAUGCUACAAGGCGCUAAGCAAUGGAAGUGAAUCUUGAGGGAAGCAAUUCUGCAGAGACACUUCCACCCUGGGAUGGCGUCGCCGGAGGACACCUACCCGUUACAGGGAAGUUGGCUGUUUGGCUGGACAACCGCAAUCUUCUUUUU'
    main()