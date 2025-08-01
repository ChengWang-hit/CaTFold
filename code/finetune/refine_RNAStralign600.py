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
from torch.cuda.amp import autocast

from utils import *

def run(model, train_loader, optimizer, train_epochs, writer):
	pos_weight = torch.Tensor([10]).cuda()
	criterion_bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean').cuda()

	print("Initialized learning rates: ")
	for group in optimizer.param_groups:
		print(f"├─ [{group['name']}] lr = {group['lr']:.2e}")
	print('time: {} start training...'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
	
	for epoch in range(train_epochs):
		loss_list = []
		bce_loss_list = []
  
		print('training...')
		for idx, train_data in enumerate(tqdm(train_loader)):
			model.train()
			node_onehot, node_pe, pad_mask, seq_length, label_ss  = train_data
			node_onehot = node_onehot.cuda()
			node_pe = node_pe.cuda()
			pad_mask = pad_mask.cuda()
			label_ss = label_ss.cuda()
			seq_length = seq_length.cuda()

			input = (node_onehot, node_pe, pad_mask, seq_length)
			with autocast(dtype=torch.bfloat16, enabled=True, cache_enabled=True):
				pred = model(input)

				# Compute loss
				bce_loss = criterion_bce(pred, label_ss)
				loss = bce_loss

			# Optimize the model
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			
			loss_list.append(loss.item())
			bce_loss_list.append(bce_loss.item())

		loss = sum(loss_list)/len(loss_list)
		bce_loss = sum(bce_loss_list) / len(bce_loss_list)
		print('time: {} epoch: {}, loss: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch+1, loss))
		print(f'bce loss: {bce_loss:.4f}')

		writer.add_scalar('train/loss', loss, epoch+1)
		writer.add_scalar('train/bce_loss', bce_loss, epoch+1)

		if epoch+1 >= 1:
			param_save = model.state_dict()
			torch.save(param_save, f'{writer.logdir}/CaTFold_epoch_{epoch+1}.pt')

def main(epoch, cuda_id=0):
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

	print('Loading dataset.')

	RNAStralign_train_data = DataGenerator(f'data/RNAStralign', f'train_max600', node_input_dim=embedding_dim_pretrain, mode='finetune', family_fold=False)
	RNAStralign_train_dataset = Dataset(RNAStralign_train_data)
	RNAStralign_train_loader = data.DataLoader(RNAStralign_train_dataset, 
									   batch_size=8, 
									   num_workers=8, 
									   pin_memory=True,
									   shuffle=True,
									   collate_fn=collate)

	print('Data Loading Done!!!')
	
	# CNN Adapter
	model = RefineNet(embedding_dim_pretrain, ct_layer_num_pretrain, nhead_pretrain, hidden_dim=128, adapter_type='cnn')
	checkpoint = torch.load(f'checkpoints/CaTFold_pretrain.pt')
	model.load_state_dict(checkpoint, strict=False)
	model.cuda()

	start = time.time()
	log_path = 'logs/finetune/RNAStralign/{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
	# record result
	writer = SummaryWriter(log_dir=log_path)

	shutil.copytree('code/finetune', log_path + '/code/finetune')

	# log file
	logger = Logger(log_path + '/output.txt')

	print("Model #Params Num : %d" % (sum([x.nelement() for x in model.parameters()]),))
	print("Adapter #Params Num : %d" % (sum([x.nelement() for x in model.predictor_adapter.parameters()]),))

	optimizer = optim.AdamW(set_lr(model))

	run(model, RNAStralign_train_loader, optimizer, epoch, writer)
	print('Running time:{}s\n'.format(time.time() - start))

def set_lr(model):
	print('Setting lr...')
	param_groups = []
	refine_lr = 1e-4
	adapter_lr = 1e-3
	
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
	main(epoch=5000, cuda_id=0)