import torch
from torch import nn
from adapters import CNNAdapter
import torch.nn.functional as F
from utils import outer_concat

class CNNBlock(nn.Module):
	'''
	CNN block for neightbor information.
	'''
	def __init__(self, embedding_dim, layer_num=3, drop_rate=0.1):
		super(CNNBlock, self).__init__()
		self.layer_num = layer_num
		self.drop_rate = drop_rate
		self.encoder = nn.ModuleList([
						nn.Sequential(
							nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1), 
							nn.GroupNorm(num_groups=8, num_channels=embedding_dim),
							nn.GELU(),
							nn.Dropout(drop_rate)
							)
						for _ in range(layer_num)])

	def forward(self, embedding):
		x = embedding.transpose(1, 2)
		for layer in self.encoder:
			x = layer(x) + x
		return x.transpose(1, 2) / len(self.encoder)

class CTBlock(nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
		super(CTBlock, self).__init__()
		# 1d-cnn
		self.cnn_block = CNNBlock(d_model)

		self.transformer_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

	def transformerencoder_attentionmaps(self, transformer_encoder, x, pad_mask):
		src_key_padding_mask = F._canonical_mask(
			mask=pad_mask,
			mask_name="src_key_padding_mask",
			other_type=F._none_or_dtype(pad_mask),
			other_name="src_mask",
			target_type=x.dtype
		)

		# _sa_block
		if transformer_encoder.norm_first:
			h = transformer_encoder.norm1(x)
			h, attn = transformer_encoder.self_attn(h, h, h,
						   attn_mask=None,
						   key_padding_mask=src_key_padding_mask,
						   need_weights=True, 
						   is_causal=False, 
						   average_attn_weights=True)
			h = transformer_encoder.dropout1(h)
			x = x + h

			# _ff_block
			x = x + transformer_encoder._ff_block(transformer_encoder.norm2(x))
		else:
			h, attn = transformer_encoder.self_attn(x, x, x,
						   attn_mask=None,
						   key_padding_mask=src_key_padding_mask,
						   need_weights=True, 
						   is_causal=False, 
						   average_attn_weights=False)
			h = transformer_encoder.dropout1(h)
			x = transformer_encoder.norm1(x + h)
			
			# _ff_block
			x = transformer_encoder.norm2(x + transformer_encoder._ff_block(x))
		
		return x, attn

	def forward(self, x, pad_mask):
		x = self.cnn_block(x)
		x, attn = self.transformerencoder_attentionmaps(self.transformer_encoder, x, pad_mask)
  
		return x, attn

class Encoder(nn.Module):
	def __init__(self, layer_num, d_model, nhead, dim_feedforward=512):
		super(Encoder, self).__init__()
		self.nhead = nhead
		self.layer_num = layer_num
		self.atten_maps = []

		self.ct_block = nn.ModuleList([CTBlock(d_model=d_model, nhead=self.nhead, dim_feedforward=dim_feedforward) for _ in range(self.layer_num)])
	
	def forward(self, node_embedding, position_embedding, pad_mask):
		embedding = torch.zeros_like(node_embedding).cuda()
		node_embedding = node_embedding + position_embedding

		for i in range(self.layer_num):
			node_embedding, attn = self.ct_block[i](node_embedding, pad_mask)
			self.atten_maps.append(attn)
			embedding += node_embedding

		return embedding / (self.layer_num)

class PredictorSS(nn.Module):
	def __init__(self, embedding_dim):
		super(PredictorSS, self).__init__()
		self.pair_encoder = nn.Sequential(
									   nn.Linear(embedding_dim * 2, embedding_dim),\
									   nn.GELU(),\
									   nn.Linear(embedding_dim, embedding_dim),\
										nn.GELU()
									)
		self.pair_out = nn.Linear(embedding_dim, 1)

	def forward(self, pair_embedding):
		pair_embedding = self.pair_encoder(pair_embedding)
		# out = self.pair_out(pair_embedding)
		# return out
		return pair_embedding

class RefineNet(nn.Module):
	def __init__(self, embedding_dim_pretrain, ct_layer_num_pretrain, nhead_pretrain, hidden_dim=64, adapter_type='cnn'):
		super(RefineNet, self).__init__()
		self.adapter_type = adapter_type

		# prior block
		self.node_embedding = nn.Linear(4, embedding_dim_pretrain)
		self.position_embedding = nn.Linear(embedding_dim_pretrain, embedding_dim_pretrain)
		self.encoder = Encoder(ct_layer_num_pretrain, embedding_dim_pretrain, nhead_pretrain, dim_feedforward=embedding_dim_pretrain)
		self.predictor_ss = PredictorSS(embedding_dim_pretrain)

		self.predictor_adapter = CNNAdapter(in_dim=embedding_dim_pretrain, conv_dim=hidden_dim)

	def pretrain_check(self, input):
		self.encoder.atten_maps.clear()
		node_onehot, node_pe, pad_mask = input
		node_embedding = self.node_embedding(node_onehot)
		position_embedding = self.position_embedding(node_pe)
		node_embedding = self.encoder(node_embedding, position_embedding, pad_mask)

		# SS
		pair_embedding = outer_concat(node_embedding, node_embedding)
		pred = self.predictor_ss.pair_out(self.predictor_ss(pair_embedding))
		pred = (pred + pred.transpose(1, 2)) / 2

		return pred.squeeze(-1) # (B, L, L)
			
	def forward(self, input):
		self.encoder.atten_maps.clear()

		node_onehot, node_pe, pad_mask, seq_length = input
  
		position_embedding = self.position_embedding(node_pe)
		node_embedding = self.node_embedding(node_onehot)
		node_embedding = self.encoder(node_embedding, position_embedding, pad_mask)
		pair_embedding = outer_concat(node_embedding, node_embedding)
		pair_embedding = self.predictor_ss(pair_embedding)
		attn_maps = torch.cat(self.encoder.atten_maps, dim=1) # (B, 64, L, L)
		input = pair_embedding, seq_length, attn_maps
		pred = self.predictor_adapter(input)

		return pred

	def inference(self, input):
		self.encoder.atten_maps.clear()

		node_onehot, node_pe, pad_mask = input
  
		position_embedding = self.position_embedding(node_pe)
		node_embedding = self.node_embedding(node_onehot)
		node_embedding = self.encoder(node_embedding, position_embedding, pad_mask)
		pair_embedding = outer_concat(node_embedding, node_embedding)
		pair_embedding = self.predictor_ss(pair_embedding)
		attn_maps = torch.cat(self.encoder.atten_maps, dim=1) # (B, 64, L, L)
		input = pair_embedding, attn_maps
		pred = self.predictor_adapter.inference(input)
		
		return pred