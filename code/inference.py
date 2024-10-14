import collections
from data_generator import DataGenerator, Dataset

import torch
from torch.utils import data
from utils import *
from config import *
from tqdm import tqdm
from network import TaGFold

def test(model, test_loader, threshold=0.5):
    print('testing……')
    model.eval()
    result = []
    result_shifted = []
    
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
                p = pred[cum:cum + seq_length[j] * seq_length[j]].reshape(seq_length[j], seq_length[j])

                p = torch.sigmoid(p)

                param = (p, node_set1[j], threshold)
                p_p = post_process_HK(param)
                # p_p = post_process_blossom(p, threshold)

                l = label[cum:cum + seq_length[j] * seq_length[j]].reshape(seq_length[j], seq_length[j])

                result.append(evaluate(p_p, l))
                result_shifted.append(evaluate_shifted(p_p, l))

                cum += seq_length[j] * seq_length[j]

    p, r, f1 = zip(*result)
    p_s, r_s, f1_s = zip(*result_shifted)

    print('precision: ', np.average(p))
    print('recall: ', np.average(r))
    print('F1: ', np.average(f1))
    print()
    print('precision(S): ', np.average(p_s))
    print('recall(S): ', np.average(r_s))
    print('F1(S): ', np.average(f1_s))
    print()

    return p, r, f1, p_s, r_s, f1_s

def main():
    args = get_args()
    config_file = args.config
    config = process_config(config_file)
    logger = Logger('results/output.txt')
    print("#####Stage 1#####")
    print('Here is the configuration of this run: ')
    print(config)

    threshold = config.threshold
    embedding_dim = config.embedding_dim
    torch.cuda.set_device(int(config.gpu_id))
    
    params = {'shuffle': False,
              'drop_last': False,
              'collate_fn': collate}
    
    RNAStralign_data_test_max600 = DataGenerator('data/RNAStralign', 'test_max600', node_input_dim=embedding_dim)
    RNAStralign_dataset_test_max600 = Dataset(RNAStralign_data_test_max600)
    RNAStralign_loader_test_max600 = data.DataLoader(RNAStralign_dataset_test_max600, 
                                 **params, 
                                 batch_size=10, 
                                 num_workers=1, 
                                 pin_memory=True)

    ArchiveII_data_max600 = DataGenerator('data/ArchiveII', 'max600', node_input_dim=embedding_dim)
    ArchiveII_dataset_max600 = Dataset(ArchiveII_data_max600)
    ArchiveII_loader_max600 = data.DataLoader(ArchiveII_dataset_max600, 
                                 **params, 
                                 batch_size=10, 
                                 num_workers=1, 
                                 pin_memory=True)
    
    RNAStralign_data_test = DataGenerator('data/RNAStralign', 'test', node_input_dim=embedding_dim)
    RNAStralign_dataset_test = Dataset(RNAStralign_data_test)
    RNAStralign_loader_test = data.DataLoader(RNAStralign_dataset_test, 
                                 **params, 
                                 batch_size=10, 
                                 num_workers=1, 
                                 pin_memory=True)
    
    ArchiveII_data = DataGenerator('data/ArchiveII', 'all', node_input_dim=embedding_dim)
    ArchiveII_dataset = Dataset(ArchiveII_data)
    ArchiveII_loader = data.DataLoader(ArchiveII_dataset, 
                                 **params, 
                                 batch_size=10, 
                                 num_workers=1, 
                                 pin_memory=True)
    print('Data Loading Done!!!')

    model = TaGFold(ct_layer_num=6)
    checkpoint_path = 'checkpoint/CaTFold_best.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")['model'])
    model.cuda()

    print('RNAStralign test......')
    precision, recall, f1, p_s, r_s, f1_s = test(model, RNAStralign_loader_test, threshold)
    
    print('ArchiveII......')
    precision, recall, f1, p_s, r_s, f1_s = test(model, ArchiveII_loader, threshold)

    print('RNAStralign_max600 test......')
    precision, recall, f1, p_s, r_s, f1_s = test(model, RNAStralign_loader_test_max600, threshold)
    
    print('ArchiveII_max600......')
    precision, recall, f1, p_s, r_s, f1_s = test(model, ArchiveII_loader_max600, threshold)

if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()