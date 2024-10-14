import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

min_val = -np.log(1e3-1)

class CTBlock(nn.Module):
	def __init__(self, embedding_dim, ct_layer_num):
		super(CTBlock, self).__init__()
		self.layer_num = ct_layer_num
		self.cnn_blocks = nn.ModuleList([CNNBlock(embedding_dim) for _ in range(self.layer_num)])
		self.trans_block = nn.ModuleList([TransformerBlock(embedding_dim) for _ in range(self.layer_num)])

	def forward(self, node_embedding, position_embedding, att_mask):
		embedding = torch.zeros_like(node_embedding).cuda()
		for i in range(self.layer_num):
			node_embedding = self.cnn_blocks[i](node_embedding)
			node_embedding = self.trans_block[i](node_embedding, position_embedding, att_mask)

			embedding += node_embedding

		return embedding / self.layer_num

class TransformerBlock(nn.Module):
	'''
	Transformer block for global information.
	'''
	def __init__(self, embedding_dim):
		super(TransformerBlock, self).__init__()
		self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=2, batch_first=True, dim_feedforward=512)])

	def forward(self, node_embedding, position_embedding, att_mask):
		embedding = node_embedding + position_embedding

		# mask pad
		# index = torch.arange(embedding.size(1)).unsqueeze(0).expand(embedding.size(0), -1).cuda()
		# pad_mask = (index >= seq_length.view(-1, 1))

		src_mask = torch.stack([att_mask, att_mask], dim=1).view(-1, att_mask.size(1), att_mask.size(2))

		out = embedding
		for mod in self.encoder:
			
			out = mod(out, src_mask=src_mask)

		return out

class CNNBlock(nn.Module):
	'''
	CNN block for neightbor information.
	'''
	def __init__(self, embedding_dim, drop_rate=0.1):
		super(CNNBlock, self).__init__()
		self.cnn1 = nn.Sequential(
							torch.nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1), 
							nn.BatchNorm1d(embedding_dim), 
							nn.ELU(),
							nn.Dropout(drop_rate))
		self.cnn2 = nn.Sequential(
							torch.nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1), 
							nn.BatchNorm1d(embedding_dim), 
							nn.ELU(),
							nn.Dropout(drop_rate))
		self.cnn3 = nn.Sequential(
							torch.nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1), 
							nn.BatchNorm1d(embedding_dim), 
							nn.ELU(),
							nn.Dropout(drop_rate))
		# self.cnn4 = nn.Sequential(
		# 					torch.nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1), 
		# 					nn.BatchNorm1d(embedding_dim), 
		# 					nn.ReLU())
		# self.cnn5 = nn.Sequential(
		# 					torch.nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1), 
		# 					nn.BatchNorm1d(embedding_dim), 
		# 					nn.ReLU())
		self.linear = nn.Linear(embedding_dim, embedding_dim)
		self.encoder = nn.ModuleList([self.cnn1, self.cnn2, self.cnn3])

	def forward(self, embedding):
		out = torch.transpose(embedding, 1, 2)
		
		for mod in self.encoder:
			out = mod(out) + out
			# out = mod(out)

		out = self.linear(torch.transpose(out, 1, 2))
		# out = torch.transpose(out, 1, 2)
		return out

class Predictor(nn.Module):
	def __init__(self, embedding_dim, drop_prob=0.1):
		super(Predictor, self).__init__()
		self.predictor = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim),\
									   nn.ELU(),\
									   nn.Dropout(drop_prob),\
									   nn.Linear(embedding_dim, 32),\
									   nn.ELU(),\
									   nn.Dropout(drop_prob),\
									   nn.Linear(32, 1))

	def forward(self, pair_embedding):
		pred = self.predictor(pair_embedding)
		return pred

class PairEncoder(nn.Module):
	def __init__(self, pair_embedding_dim, layer_num):
		super(PairEncoder, self).__init__()
		self.layer_num = layer_num
		self.pair_encoder = nn.ModuleList([PairEncoderBlock(pair_embedding_dim) for _ in range(layer_num)])


	def forward(self, cm_conved, mask):
		out = cm_conved
		for i in range(self.layer_num):
			cm_conved = self.pair_encoder[i](cm_conved, mask)
			out = torch.cat((out, cm_conved), dim=1)

		return out

class PairEncoderBlock(nn.Module):
	def __init__(self, pair_embedding_dim):
		super(PairEncoderBlock, self).__init__()
		self.gamma = nn.Parameter(torch.randn((3,3)))
		self.bias = nn.Parameter(torch.randn(1))

		self.row_scale = nn.Parameter(torch.randn(1))
		self.row_bias = nn.Parameter(torch.randn(1))
		self.col_scale = nn.Parameter(torch.randn(1))
		self.col_bias = nn.Parameter(torch.randn(1))
		self.pos = torch.Tensor([[0, 0, 1],
								 [0, 1, 0],
								 [1, 0, 0]])
		self.register_buffer('a_diag', self.pos)

	def forward(self, cm, mask):
		cm_conved = F.relu(cm)
		kernel = self.a_diag / (3 + torch.abs(self.gamma)).unsqueeze(dim=0).unsqueeze(dim=0)
		cm_conved_a_diag = F.relu(F.conv2d(input=cm_conved, weight=kernel, bias=self.bias, padding=(1, 1)))
		
		seq_length = cm.size(2)
		cm_row = (-1 / (seq_length + torch.abs(self.row_scale))) * (-cm_conved + cm_conved.sum(dim=3, keepdim=True)) + (self.row_bias)
		cm_col = (-1 / (seq_length + torch.abs(self.col_scale))) * (-cm_conved + cm_conved.sum(dim=2, keepdim=True)) + (self.col_bias)
		cm = cm + cm_conved_a_diag + (cm_row + torch.transpose(cm_row, 2, 3) + cm_col + torch.transpose(cm_col, 2, 3)) / 4

		return  cm * mask

class TaGFold(nn.Module):
	def __init__(self, embedding_dim=128, ct_layer_num=6):
		super(TaGFold, self).__init__()
		self.embedding_dim = embedding_dim
		self.pair_embedding_dim = 1

		self.node_embedding = nn.Linear(4, embedding_dim)
		self.position_embedding = nn.Linear(embedding_dim, embedding_dim)

		# CT Block
		self.ct_embedding = CTBlock(embedding_dim, ct_layer_num)

		pair_encoder_layer_num = 2
		self.pair_encoder = PairEncoder(self.pair_embedding_dim, pair_encoder_layer_num)

		# predictor
		self.predictor = Predictor(self.embedding_dim)

		self.integ = nn.Parameter(torch.randn((pair_encoder_layer_num+1, 1)))

	# reshape contact map
	def restore_pred(self, cm_raw, constrained_index, constrained_index_length, seq_length):
		contact_maps_conved = []

		cum_index = 0
		cum_offset = 0
		for i in range(len(seq_length)):
			c_idx = constrained_index[:, cum_index:cum_index+constrained_index_length[i]] - cum_offset
			cm = torch.zeros((seq_length[i], seq_length[i])).cuda()
			cm[c_idx[0, :], c_idx[1, :]] = cm_raw[cum_index:cum_index+c_idx.size(1)].view(-1)

			mask = torch.zeros((seq_length[i], seq_length[i])).cuda()
			mask[c_idx[0, :], c_idx[1, :]] = 1

			cm = (cm + cm.T) / 2 * mask
			# contact_maps.append((cm + (1-mask)*min_val).reshape(-1))

			cm_conved = cm.unsqueeze(dim=0).unsqueeze(dim=0)

			cm = self.pair_encoder(cm_conved, mask).permute(0, 2, 3, 1)
			cm = cm @ (torch.softmax(input=self.integ, dim=0))
			cm = cm.squeeze(0).squeeze(-1)

			cm = (cm+cm.T) / 2 * mask
			cm = cm + (1-mask)*min_val
			contact_maps_conved.append(cm.reshape(-1))

			cum_index += constrained_index_length[i]
			cum_offset += seq_length[i]

		return torch.cat(contact_maps_conved, dim=0)

	def forward(self, input):
		node_onehot, trans_pe, constrained_index, constrained_index_length, seq_length, label_index, att_mask = input

		node_embedding = self.node_embedding(node_onehot)
		position_embedding = self.position_embedding(trans_pe)

		node_embedding = self.ct_embedding(node_embedding, position_embedding, att_mask)

		# non-batch format
		seq_length_list = seq_length.tolist()
		node_embedding = torch.cat([node_embedding[i][:seq_length[i]] for i in range(len(seq_length_list))], dim=0)

		cm_raw = self.predictor(torch.cat((node_embedding[constrained_index[0, :]], node_embedding[constrained_index[1, :]]), dim=1))

		return self.restore_pred(cm_raw, constrained_index, constrained_index_length, seq_length)