import torch
from torch.utils import data
from torch.nn import functional as F
from tqdm import tqdm

from config import *
from data_generator import DataGenerator, Dataset
from network import RefineNet
from utils import *

def test(model, data_loader, threshold=0.5):
    model.eval()
    result = []
    result_s = []
    f1_list = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            node_onehot, node_pe, pad_mask, mask_matrix, contact_map, seq_length, node_set1_list = data
            node_onehot = node_onehot.cuda()
            node_pe = node_pe.cuda()
            pad_mask = pad_mask.cuda()
            mask_matrix = mask_matrix.cuda()
            contact_map = contact_map.cuda()
            seq_length = seq_length.cuda()

            input = (node_onehot, node_pe, pad_mask)

            pred = model.inference(input)
   
            # retain high probability
            base_pair_prob = F.sigmoid(pred) * mask_matrix

            # postprocessing_params = []
            for i in range(len(seq_length)):
                contact_map_pred = base_pair_prob[i, :seq_length[i], :seq_length[i]]
                contact_map_pred = post_process_argmax(contact_map_pred, threshold)
                # param = (contact_map_pred, node_set1_list[i], threshold)
                # contact_map_pred = post_process_HK(param)
                # contact_map_pred = post_process_maximum_weight_matching(param)

                contact_map_label = contact_map[i, :seq_length[i], :seq_length[i]]

                result.append(evaluate(contact_map_pred, contact_map_label))
                result_s.append(evaluate_shifted(contact_map_pred, contact_map_label))

                f1_list.append(result[-1][-1])
    
    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result)
    print('precision: ', np.average(nt_exact_p))
    print('recall: ', np.average(nt_exact_r))
    print('F1: ', np.average(nt_exact_f1))
    print('std error f1: ', np.std(nt_exact_f1, ddof=1) / np.sqrt(len(nt_exact_f1)))
    print()

    # nt_exact_p_s,nt_exact_r_s,nt_exact_f1_s = zip(*result_s)
    # print('precision_s: ', np.average(nt_exact_p_s))
    # print('recall_s: ', np.average(nt_exact_r_s))
    # print('F1_s: ', np.average(nt_exact_f1_s))
    # print()

    return np.average(nt_exact_f1), np.average(nt_exact_p), np.average(nt_exact_r)

def main(cuda_idx=0):
    args = get_args()
    config_file = args.config
    config = process_config(config_file)
    print('Here is the configuration of this run: ')
    print(config)
    
    torch.cuda.set_device(cuda_idx)
    embedding_dim_pretrain = config.embedding_dim_pretrain
    ct_layer_num_pretrain = config.layer_num_pretrain
    nhead_pretrain = config.nhead_pretrain
    # batch_size = 16
    # num_workers = 8
    
    params = {'drop_last': False,
              'collate_fn': collate_test}

    RNAStralign_test_data = DataGenerator(f'data/RNAStralign', f'test', node_input_dim=embedding_dim_pretrain)
    RNAStralign_test_dataset = Dataset(RNAStralign_test_data)
    RNAStralign_test_loader = data.DataLoader(RNAStralign_test_dataset, 
    								   **params, 
    								   batch_size=1,
    								   num_workers=4, 
    								   pin_memory=True,
    								   shuffle=False)
    
    ArchiveII_data_max600 = DataGenerator(f'data/ArchiveII', f'all_max600', node_input_dim=embedding_dim_pretrain)
    ArchiveII_dataset_max600 = Dataset(ArchiveII_data_max600)
    ArchiveII_loader_max600 = data.DataLoader(ArchiveII_dataset_max600, 
    								   **params, 
    								   batch_size=1,
    								   num_workers=4, 
    								   pin_memory=True,
    								   shuffle=False)
    
    bpRNA1m_test_data = DataGenerator(f'data/bpRNA_1m', f'test', node_input_dim=embedding_dim_pretrain)
    bpRNA1m_test_dataset = Dataset(bpRNA1m_test_data)
    bpRNA1m_test_loader = data.DataLoader(bpRNA1m_test_dataset, 
                                       **params, 
                                       batch_size=1,
                                       num_workers=8, 
                                       pin_memory=True,
                                       shuffle=False)
    
    bpRNAnew_data = DataGenerator(f'data/bpRNA_new', f'all', node_input_dim=embedding_dim_pretrain)
    bpRNAnew_dataset = Dataset(bpRNAnew_data)
    bpRNAnew_loader = data.DataLoader(bpRNAnew_dataset, 
    								   **params, 
    								   batch_size=1,
    								   num_workers=4, 
    								   pin_memory=True,
    								   shuffle=False)
    
    ArchiveII_families = ['5s', 'srp', 'tRNA', 'tmRNA', 'RNaseP', 'grp1', '16s', 'telomerase', '23s']

    family_loaders = [data.DataLoader(
                        Dataset(DataGenerator(f'data/ArchiveII/family_fold', f'{family}', node_input_dim=embedding_dim_pretrain, mode='test', family_fold=True)),
                        **params, 
                        batch_size=1,
                        num_workers=4, 
                        pin_memory=True,
                        shuffle=False)
                        for family in ArchiveII_families]

    print('Data Loading Done!!!')
    model = RefineNet(embedding_dim_pretrain, ct_layer_num_pretrain, nhead_pretrain, hidden_dim=128, adapter_type='cnn')
    model.cuda()
    print("Model #Params Num : %d" % (sum([x.nelement() for x in RefineNet(embedding_dim_pretrain, ct_layer_num_pretrain, nhead_pretrain).parameters()]),))

    # print('RNAStralign test...')
    # checkpoint = torch.load(f'checkpoints/CaTFold_finetune_RNAstraligntest.pt', map_location=f'cuda:{cuda_idx}')
    # model.load_state_dict(checkpoint, strict=False)
    # test(model, RNAStralign_test_loader)

    print('ArchiveII max600...')
    checkpoint = torch.load(f'checkpoints/CaTFold_finetune_ArchiveII600.pt', map_location=f'cuda:{cuda_idx}')
    model.load_state_dict(checkpoint, strict=False)
    test(model, ArchiveII_loader_max600)

    print('bpRNA-1m test...')
    checkpoint = torch.load('checkpoints/CaTFold_finetune_bpRNA1mtest.pt')
    model.load_state_dict(checkpoint, strict=False)
    test(model, bpRNA1m_test_loader)

    print('bpRNAnew...')
    checkpoint = torch.load(f'checkpoints/CaTFold_finetune_bpRNAnew_withaug.pt')
    model.load_state_dict(checkpoint, strict=False)
    test(model, bpRNAnew_loader)

    print('ArchiveII families...')
    mean_f1 = 0
    for idx, family in enumerate(ArchiveII_families):
        print(f'{family}...')
        checkpoint = torch.load(f'checkpoints/CaTFold_finetune_{family}.pt', map_location=f'cuda:{cuda_idx}')
        model.load_state_dict(checkpoint, strict=False)
        mean_f1 += test(model, family_loaders[idx])[0]
        
    mean_f1 /= len(ArchiveII_families)
    print('Mean F1 for ArchiveII families:', mean_f1)

if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    main(cuda_idx=0)