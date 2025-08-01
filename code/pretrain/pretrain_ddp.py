import os

import datetime

import shutil
import time

import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils import data
from torch.nn import functional as F
from tqdm import tqdm

from config import *
from data_generator import DataGenerator, Dataset
from network import PretrainNet
from utils import *
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_

import os

import gc

def init_ddp():
	os.environ["MASTER_ADDR"] = "localhost" # single machine
	os.environ["MASTER_PORT"] = "2025"     # port
	dist.init_process_group(backend="nccl")
	torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

# random
def get_ddp_generator(seed=2024):
	'''
	Different processes use different random seeds
	'''
	local_rank = dist.get_rank()
	g = torch.Generator()
	g.manual_seed(seed+local_rank)
	return g

def run(local_rank, model, train_loader, optimizer, train_epochs, writer, threshold, accumulation_steps):
	pos_weight = torch.Tensor([10]).cuda()
	criterion_bce_ss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum').cuda()

	if local_rank == 0:
		print("Initialized learning rate: ", optimizer.param_groups[0]['lr'])
		print('time: {} start training...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

	for epoch in range(train_epochs):
		loss_list = []
		loss_ss_list = []
		batch_loss_list = []
		print('training...')
		train_loader.sampler.set_epoch(epoch)
		for idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
			model.train()
			# Skip checkpoint
			# if epoch == 0 and idx+1 <= 170820:
			# 	continue
				
			node_onehot, node_pe, pred_pairs, pad_mask, seq_length, label_ss  = data
			node_onehot = node_onehot.cuda(non_blocking=True)
			node_pe = node_pe.cuda(non_blocking=True)
			pred_pairs = pred_pairs.cuda(non_blocking=True)
			pad_mask = pad_mask.cuda(non_blocking=True)
			seq_length = seq_length.cuda(non_blocking=True)
			label_ss = label_ss.cuda(non_blocking=True)

			input = (node_onehot, node_pe, pred_pairs, pad_mask)
			
			with autocast(dtype=torch.bfloat16, enabled=True, cache_enabled=True):
				pred_ss = model(input)
			
				# Compute loss
				loss_ss = (criterion_bce_ss(pred_ss, label_ss) / len(seq_length)) / accumulation_steps

				loss = loss_ss

			loss.backward()
			if (idx + 1) % accumulation_steps == 0:
				clip_grad_norm_(model.parameters(), max_norm=1.0)
				# Optimize the model
				optimizer.step()
				optimizer.zero_grad(set_to_none=True)
							
			loss_list.append(loss.item() * accumulation_steps)
			loss_ss_list.append(loss_ss.item() * accumulation_steps)
			batch_loss_list.append(loss.item() * accumulation_steps)
		
			# save model
			if (idx+1) % (len(train_loader) // 30) == 0:
				if local_rank == 0:
					print(f'Batch #{idx+1}')
					print(f'ss loss: {np.mean(loss_ss_list):.4f}')
					print(f'total loss: {np.mean(loss_list):.4f}')
					print(f'batch loss: {np.mean(batch_loss_list):.4f}')
					print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
					writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch*len(train_loader)+idx+1)
					writer.add_scalar('train/batch_loss',np.mean(batch_loss_list) , epoch*len(train_loader)+idx+1)

					param_save = {
					'model':model.module.state_dict(),
					'optimizer':optimizer.state_dict(),
					}
					torch.save(param_save,  f'{writer.logdir}/Pretrain_{epoch+1}_{idx+1}.pt')

					batch_loss_list = []
				
		loss = torch.tensor(np.mean(loss_list)).cuda(local_rank)
		dist.all_reduce(loss, op=dist.ReduceOp.AVG)
		
		if local_rank == 0:
			loss_ss = np.mean(loss_ss_list)
			print('time: {} epoch: {}, loss: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch+1, loss))
			print(f'ss loss: {loss_ss:.4f}')
			print(f'total loss: {loss:.4f}')

			writer.add_scalar('train/loss', loss, epoch+1)
			writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch+1)
			writer.add_scalar('train/loss_ss', loss_ss, epoch+1)

			param_save = {
				'model':model.module.state_dict(),
				'optimizer':optimizer.state_dict(),
			}
			torch.save(param_save,  f'{writer.logdir}/Pretrain_{epoch+1}_finished.pt')
			gc.collect()

def main():
	args = get_args()
	config_file = args.config
	config = process_config(config_file)
	print('Here is the configuration of this run: ')
	print(config)

	init_ddp()
	local_rank = dist.get_rank()
	seed_torch(2025)
	train_epochs = config.train_epochs
	threshold = config.threshold
	embedding_dim = config.embedding_dim
	layer_num = config.layer_num
	nhead = config.nhead
	batch_size = 48
	num_workers = 12

	print('Loading dataset.')
	train_data = DataGenerator(f'data/data_for_pretrain', f'Contrafold_seq_ss_21M', node_input_dim=embedding_dim)
	train_dataset = Dataset(train_data)
	train_sampler = DistributedSampler(train_dataset, shuffle=True)
	g = get_ddp_generator()
	train_loader = data.DataLoader(train_dataset, 
									batch_size=batch_size, 
									num_workers=num_workers, 
									sampler=train_sampler, 
									pin_memory=True, 
									generator=g,
									collate_fn=collate)

	print('Data Loading Done!!!')
 
	model = PretrainNet(embedding_dim, layer_num, nhead)
	model.cuda()
	# model = torch.compile(model)
	model = DDP(model, device_ids=[local_rank], output_device=local_rank)
	writer = None

	if local_rank == 0:
		start = time.time()
		log_path = 'logs/pretrain/{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
		# record result
		writer = SummaryWriter(log_dir=log_path)

		# Record the code
		shutil.copytree('code/pretrain', log_path + '/code/pretrain')

		# log file
		logger = Logger(log_path + '/output.txt')

		print("Model #Params Num : %d" % (sum([x.nelement() for x in model.parameters()]),))

	optimizer = optim.AdamW(model.parameters(), lr=1e-4)

	run(local_rank, model, train_loader, optimizer, train_epochs, writer, threshold, accumulation_steps=6)
	print('Running time:{}s\n'.format(time.time() - start))
	dist.destroy_process_group()

	print(f'pretrain finished!')

if __name__ == '__main__':
	"""
	See module-level docstring for a description of the script.
	"""
	main()
	# Running in the terminal: CUDA_VISIBLE_DEVICES=0,1 your_torchrun_path --standalone --nproc_per_node=2 pretrain_ddp.py