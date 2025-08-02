import torch
from torch.utils import data
from torch.nn import functional as F
from tqdm import tqdm

from config import *
from data_generator import DataGenerator, Dataset
from network import RefineNet
from utils import *
import multiprocessing
import time

def inference_st(model, data_loader, threshold=0.5):
    model.eval()
    pred_contactmaps = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            node_onehot, node_pe, pad_mask, mask_matrix, seq_length, node_set1_list = data
            node_onehot = node_onehot.cuda()
            node_pe = node_pe.cuda()
            pad_mask = pad_mask.cuda()
            mask_matrix = mask_matrix.cuda()
            seq_length = seq_length.cuda()

            input = (node_onehot, node_pe, pad_mask)

            start_time = time.time()
            
            pred = model(input)

            # retain high probability
            base_pair_prob = F.sigmoid(pred) * mask_matrix

            # postprocessing_params = []
            for i in range(len(seq_length)):
                contact_map_pred = base_pair_prob[i, :seq_length[i], :seq_length[i]]
                # contact_map_pred = post_process_argmax(contact_map_pred, threshold)
                param = (contact_map_pred.cpu().numpy(), node_set1_list[i], threshold)
                # contact_map_pred = post_process_HK(param)
                contact_map_pred = post_process_maximum_weight_matching(param)
            
                # pred_contactmaps.append(contact_map_pred.astype(bool))
            
    return pred_contactmaps

def inference_mt(model, data_loader, threshold=0.5):
    pool_workers = min(data_loader.batch_size, multiprocessing.cpu_count())
    # pool_workers = 4
    model.eval()
    pred_contactmaps = []
    with multiprocessing.Pool(processes=pool_workers) as pool:
        with torch.no_grad():
            for data in tqdm(data_loader):
                node_onehot, node_pe, pad_mask, mask_matrix, seq_length, node_set1_list = data
                node_onehot = node_onehot.cuda()
                node_pe = node_pe.cuda()
                pad_mask = pad_mask.cuda()
                mask_matrix = mask_matrix.cuda()
                seq_length = seq_length.cuda()

                input = (node_onehot, node_pe, pad_mask)
                
                pred = model(input)

                # retain high probability
                base_pair_prob = F.sigmoid(pred) * mask_matrix

                postprocessing_params = []
                for i in range(len(seq_length)):
                    contact_map_pred = base_pair_prob[i, :seq_length[i], :seq_length[i]]
                    # contact_map_pred = post_process_argmax(contact_map_pred, threshold)
                    param = (contact_map_pred.cpu().numpy(), node_set1_list[i], threshold)
                    # contact_map_pred = post_process_HK(param)
                    postprocessing_params.append(param)
                
                # batch_contact_maps = pool.map(post_process_HK, postprocessing_params)
                batch_contact_maps = pool.map(post_process_maximum_weight_matching, postprocessing_params)
                pred_contactmaps.extend([cm.astype(bool) for cm in batch_contact_maps])

    return pred_contactmaps

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
    fasta_path = config.fasta_path
    checkpoint_path = config.checkpoint_path
    
    params = {'drop_last': False,
              'collate_fn': collate}
    
    inferdata = DataGenerator(fasta_path, node_input_dim=embedding_dim_pretrain)
    dataset = Dataset(inferdata)
    dataloader = data.DataLoader(dataset, 
                                       **params, 
                                       batch_size=2, 
                                       num_workers=2, 
                                       pin_memory=True,
                                       shuffle=False)

    print('Data Loading Done!!!')
    model = RefineNet(embedding_dim_pretrain, ct_layer_num_pretrain, nhead_pretrain, hidden_dim=128, adapter_type='cnn')
    model.cuda()
    print("Model #Params Num : %d" % (sum([x.nelement() for x in RefineNet(embedding_dim_pretrain, ct_layer_num_pretrain, nhead_pretrain).parameters()]),))

    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{cuda_idx}')
    model.load_state_dict(checkpoint, strict=False)

    start = time.time()
    # contact_maps = inference_st(model, dataloader)
    contact_maps = inference_mt(model, dataloader)
    
    print(f"Total Time: {time.time() - start:.2f} seconds")
    
    save_results(contact_maps, inferdata, config.output_dir)

if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    main(cuda_idx=0)