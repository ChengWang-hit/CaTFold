import torch
from torch import nn

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
	def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True):
		super(CTBlock, self).__init__()
		# 1d-cnn
		self.cnn_block = CNNBlock(d_model)
		self.transformer_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first)

	def forward(self, x, pad_mask):
		x = self.cnn_block(x)
		x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
		return x

class Encoder(nn.Module):
	def __init__(self, layer_num, d_model, nhead, dim_feedforward=512):
		super(Encoder, self).__init__()
		self.nhead = nhead
		self.layer_num = layer_num

		self.ct_block = nn.ModuleList([CTBlock(d_model=d_model, nhead=self.nhead, dim_feedforward=dim_feedforward, batch_first=True) for _ in range(self.layer_num)])
	
	def forward(self, node_embedding, position_embedding, pad_mask):
		embedding = torch.zeros_like(node_embedding).cuda()
		node_embedding = node_embedding + position_embedding

		for i in range(self.layer_num):
			node_embedding = self.ct_block[i](node_embedding, pad_mask)
			embedding += node_embedding

		return embedding / (self.layer_num)

class PredictorSS(nn.Module):
	def __init__(self, embedding_dim, drop_prob=0.1):
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
		out = self.pair_out(pair_embedding)
		return out

class PretrainNet(nn.Module):
	def __init__(self, embedding_dim, layer_num, nhead):
		super(PretrainNet, self).__init__()

		self.node_embedding = nn.Linear(4, embedding_dim)
		self.position_embedding = nn.Linear(embedding_dim, embedding_dim)

		self.encoder = Encoder(layer_num, embedding_dim, nhead, dim_feedforward=embedding_dim)

		self.predictor_ss = PredictorSS(embedding_dim)

	def forward(self, input):
		node_onehot, node_pe, pred_pairs, pad_mask = input

		B, L, _ = node_onehot.shape # B, L

		node_embedding = self.node_embedding(node_onehot)
		position_embedding = self.position_embedding(node_pe)
		node_embedding = self.encoder(node_embedding, position_embedding, pad_mask)

		valid_mask = ~pad_mask
		node_embedding = node_embedding[valid_mask]

		emb_i = node_embedding[pred_pairs[0, :]]
		emb_j = node_embedding[pred_pairs[1, :]]
  
		# SS
		cm_ur = self.predictor_ss(torch.cat((emb_i, emb_j), dim=1))
		cm_bl = self.predictor_ss(torch.cat((emb_j, emb_i), dim=1))
		cm = (cm_ur + cm_bl) / 2
		pred_ss = cm.view(-1)

		return pred_ss.view(-1)

	def inference(self, input):
		node_onehot, node_pe, pred_pairs, pad_mask = input
		position_embedding = self.position_embedding(node_pe)
		
		# SS
		node_embedding = self.node_embedding(node_onehot)
		node_embedding = self.encoder(node_embedding, position_embedding, pad_mask)
		valid_mask = ~pad_mask
		node_embedding = node_embedding[valid_mask]
		cm_ur = self.predictor_ss(torch.cat((node_embedding[pred_pairs[0, :]], node_embedding[pred_pairs[1, :]]), dim=1))
		cm_bl = self.predictor_ss(torch.cat((node_embedding[pred_pairs[1, :]], node_embedding[pred_pairs[0, :]]), dim=1))
		cm = (cm_ur + cm_bl) / 2
		pred_ss = cm

		return pred_ss.view(-1)