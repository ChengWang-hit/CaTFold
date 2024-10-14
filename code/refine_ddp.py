import os

import datetime

import collections
import shutil
import time

import torch
# from line_profiler import LineProfiler
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils import data
from tqdm import tqdm

from config import *
from data_generator import DataGenerator, Dataset
from network import TaGFold
from utils import *
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

RNA_SS_data = collections.namedtuple('RNA_SS_data', 
            'seq ss_label length name pairs')

def init_ddp():
    os.environ["MASTER_ADDR"] = "localhost" # single machine
    os.environ["MASTER_PORT"] = "2025"      # port
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

# random
def get_ddp_generator(seed=2024):
    '''
    Different processes use different random seeds
    '''
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt

def run(local_rank, model, refine_loader_all, optimizer, scheduler, epoches, writer):
    pos_weight = torch.Tensor([10]).cuda()
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()
    print("Initial lr: ", optimizer.param_groups[0]['lr'])

    if local_rank == 0:
        print('time: {} start training...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    for epoch in range(epoches):
        loss_list = []
        bce_loss_list = []
        model.train()
        if local_rank == 0:
            print('training……')

        # print('step all')
        refine_loader_all.sampler.set_epoch = epoch
        for data in tqdm(refine_loader_all):
            node_onehot, trans_pe, constrained_index, constrained_index_length, label, seq_length, label_index, _, att_mask = data
            node_onehot = node_onehot.cuda()
            trans_pe = trans_pe.cuda()
            label = label.cuda()
            label_index = label_index.cuda()
            att_mask = att_mask.cuda()

            input = (node_onehot, trans_pe, constrained_index, constrained_index_length, seq_length, label_index, att_mask)
            pred = model(input)

            bce_loss = criterion_bce_weighted(pred, label)
            loss = bce_loss

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            bce_loss_list.append(bce_loss.item())
        
        if (epoch+1) % 2 == 0:
            scheduler.step()

        if local_rank == 0:
            loss = sum(loss_list) / len(loss_list)
            bce_loss = sum(bce_loss_list) / len(bce_loss_list)

            print('time: {} epoch: {}, lr: {}, loss: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                                epoch+1, optimizer.param_groups[0]['lr'], loss))
            print(f'bce loss: {bce_loss:.9f}')

            writer.add_scalar('train/loss', loss, epoch+1)
            writer.add_scalar('train/bce_loss', bce_loss, epoch+1)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch+1)

        if epoch+1 > 0:
            if local_rank == 0:
                param_save = {
                'model':model.module.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict()
                }
                torch.save(param_save,  f'{writer.logdir}/CaTFold_refine_{epoch+1}.pt')

def main(config, pretrained_model_path):
    init_ddp()
    local_rank = dist.get_rank()

    refine_epochs = config.refine_epochs
    threshold = config.threshold
    embedding_dim = config.embedding_dim
    # torch.set_num_threads(8)
    
    params = {'drop_last': False,
            'collate_fn': collate}

    refine_data_all = DataGenerator(f'data/RNAStralign', f'train_filtered', node_input_dim=embedding_dim)
    refine_dataset_all = Dataset(refine_data_all)
    refine_sampler_all = DistributedSampler(refine_dataset_all, shuffle=True)
    g = get_ddp_generator()
    refine_loader_all = data.DataLoader(refine_dataset_all, 
                                       **params, 
                                       batch_size=10, 
                                       num_workers=5, 
                                       sampler=refine_sampler_all, 
                                       pin_memory=True, 
                                       generator=g)
    if local_rank == 0:
        print('Data Loading Done!!!')

    model = TaGFold(embedding_dim=embedding_dim, ct_layer_num=6)

    param = torch.load(pretrained_model_path)
    model.load_state_dict(param['model'])

    model.cuda()
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if local_rank == 0:
        start = time.time()

        log_path = 'logs/refine/{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        # record result
        writer = SummaryWriter(log_dir=log_path)

        # record important files
        shutil.copyfile('code/network.py', log_path + '/network_refine.py')
        shutil.copyfile('code/data_generator.py', log_path + '/data_generator.py')
        shutil.copyfile('code/refine_ddp.py', log_path + '/refine_ddp.py')
        shutil.copyfile('code/utils.py', log_path + '/utils.py')
        shutil.copyfile('code/config.json', log_path + '/config.json')

        # log file
        logger = Logger(log_path + '/output.txt')

        print("Model #Params Num : %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
    
    else: 
        writer = None

    optimizer_refine = optim.AdamW(params=model.parameters(), lr=1e-4)
    scheduler_refine = optim.lr_scheduler.ExponentialLR(optimizer_refine, gamma=0.99)
    
    print('refine!')
    run(local_rank, model, refine_loader_all, optimizer_refine, scheduler_refine, refine_epochs, writer)
    
    if local_rank == 0:
        print('Running time:{}s'.format(time.time() - start))
    dist.destroy_process_group()

if __name__ == '__main__':
    args = get_args()
    config_file = args.config
    config = process_config(config_file)
    print('Here is the configuration of this run: ')
    print(config)

    '''
    Specifies the pretrained models directory
    '''
    pretrained_model_path = 'checkpoint/CaTFold_best.pt'
    main(config, pretrained_model_path)
    # Running in the terminal: CUDA_VISIBLE_DEVICES=0,1 /home/wangcheng/anaconda3/envs/ufold/bin/torchrun --standalone --nproc_per_node=2 code/refine_ddp.py
