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
from network import RefineNet

from utils import *

def run(model, train_loader, optimizer, train_epochs, writer):
	pos_weight = torch.Tensor([5]).cuda()
	criterion_bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum').cuda()

	print("Initialized learning rates: ")
	for group in optimizer.param_groups:
		print(f"├─ [{group['name']}] lr = {group['lr']:.2e}")
	print('time: {} start training...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

	global_idx = 0
	for epoch in range(train_epochs):
		loss_list = []
		bce_loss_list = []

		print('training...')
		model.train()
		for idx, train_data in enumerate(tqdm(train_loader)):
			global_idx += 1
			# train distribution
			node_onehot, node_pe, pad_mask, seq_length, label_ss  = train_data
			node_onehot = node_onehot.cuda()
			node_pe = node_pe.cuda()
			pad_mask = pad_mask.cuda()
			label_ss = label_ss.cuda()
			seq_length = seq_length.cuda()

			input = (node_onehot, node_pe, pad_mask, seq_length)

			pred = model(input)

			# Compute loss
			bce_loss = criterion_bce(pred, label_ss) / len(seq_length)
	
			loss = bce_loss

			# Optimize the model
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			loss_list.append(loss.item())
			bce_loss_list.append(bce_loss.item())
		
		loss = sum(loss_list)/len(loss_list)
		bce_loss = sum(bce_loss_list) / len(bce_loss_list)

		print('time: {} epoch: {}, loss: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
												epoch+1, loss))
		print(f'bce loss: {bce_loss:.4f}')

		writer.add_scalar('train/loss', loss, epoch+1)
		writer.add_scalar('train/bce_loss', bce_loss, epoch+1)
		
		for param_group in optimizer.param_groups:
			lr = param_group['lr']
			group_name = param_group.get('name', 'unnamed_group')
			writer.add_scalar(
				f'train/lr/{group_name}', 
				lr, 
				epoch+1
			)

		if epoch+1 >= 1:
			param_save = model.state_dict()
			torch.save(param_save, f'{writer.logdir}/CaTFold_epoch_{epoch+1}.pt')

def main(test_family, epoch, cuda_id=0):
	args = get_args()
	config_file = args.config
	config = process_config(config_file)
	print('Here is the configuration of this run: ')
	print(config)
	
	embedding_dim_pretrain = config.embedding_dim_pretrain
	ct_layer_num_pretrain = config.layer_num_pretrain
	nhead_pretrain = config.nhead_pretrain
	torch.cuda.set_device(cuda_id)
	seed_torch(2025)
	batch_size = 8
	num_workers = 8

	print('Loading dataset.')
 
	train_data = None
	for family in families:
		if family != test_family:
			if train_data is None:
				train_data = DataGenerator(f'data/ArchiveII/family_fold', f'{family}', node_input_dim=embedding_dim_pretrain, mode='finetune', family_fold=True)
			else:
				tmp_data = DataGenerator(f'data/ArchiveII/family_fold', f'{family}', node_input_dim=embedding_dim_pretrain, mode='finetune', family_fold=True)
				train_data.merge(tmp_data)

	train_dataset = Dataset(train_data)
	train_loader = data.DataLoader(train_dataset, 
									   collate_fn=collate, 
									   batch_size=batch_size, 
									   num_workers=num_workers, 
									   shuffle=True,
									   pin_memory=True)

	print('Data Loading Done!!!')
	
	# CNN Adapter
	model = RefineNet(embedding_dim_pretrain, ct_layer_num_pretrain, nhead_pretrain, hidden_dim=128)
	checkpoint = torch.load(f'checkpoints/CaTFold_pretrain.pt')
	model.load_state_dict(checkpoint, strict=False)
	model.cuda()
	
	print("Model #Params Num : %d" % (sum([x.nelement() for x in model.parameters()]),))
	print("Adapter #Params Num : %d" % (sum([x.nelement() for x in model.predictor_adapter.parameters()]),))

	start = time.time()
	log_path = 'logs/finetune/family_fold/{}_{}_PK_F1'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), test_family)
	# record result
	writer = SummaryWriter(log_dir=log_path)

	shutil.copytree('code/finetune', log_path + '/code/finetune')

	# log file
	logger = Logger(log_path + '/output.txt')

	optimizer = optim.AdamW(set_lr(model))

	run(model, train_loader, optimizer, epoch, writer)
	print('Running time:{}s\n'.format(time.time() - start))

def set_lr(model):
	print('Setting lr...')
	param_groups = []
	refine_lr = 1e-6
	adapter_lr = 1e-4
	
	# Set learning rates for modules to be fine-tuned
	node_embedding_params = {
			"params": model.node_embedding.parameters(),
			"lr": refine_lr,
			"name": "node_embedding"
		}
	param_groups.append(node_embedding_params)
	
	position_embedding_params = {
			"params": model.position_embedding.parameters(),
			"lr": refine_lr,
			"name": "position_embedding"
		}
	param_groups.append(position_embedding_params)

	encoder_layers = model.encoder.ct_block
	# Initialize learning rates layer-wise
	lr_scales = [refine_lr for i in range(8)]
	for layer_idx, (module, base_lr) in enumerate(zip(encoder_layers, lr_scales)):
		layer_params = {
			"params": module.parameters(),
			"lr": base_lr,
			"name": f"ct_block_{layer_idx+1}"
		}
		param_groups.append(layer_params)
	
	predictor_params = {
		"params": model.predictor_ss.parameters(),
		"lr": refine_lr,
		"name": "predictor_ss"
	}
	param_groups.append(predictor_params)

	# Set learning rate for the adapter
	adapter_params = {
		"params": model.predictor_adapter.parameters(),
		"lr": adapter_lr,
		"name": "predictor_adapter"
	}
	param_groups.append(adapter_params)

	return param_groups

if __name__ == '__main__':
	"""
	See module-level docstring for a description of the script.
	"""
	
	families = ['5s', 'srp', 'tRNA', 'tmRNA', 'RNaseP', 'grp1', '16s', 'telomerase', '23s']
 
	for test_family in families:
		print(f'test_family: {test_family}')
		main(test_family, 50, cuda_id=0)