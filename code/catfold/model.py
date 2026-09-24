import torch
import torch.nn.functional as F
from torch import nn

from .adapters import CaTFoldAdapter


class CNNBlock(nn.Module):
    def __init__(self, embedding_dim, layer_num=3, drop_rate=0.1):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=embedding_dim),
                    nn.GELU(),
                    nn.Dropout(drop_rate),
                )
                for _ in range(layer_num)
            ]
        )

    @staticmethod
    def _masked_group_norm(x, norm, valid_mask):
        if valid_mask is None:
            return norm(x)
        batch, channels, length = x.shape
        groups = norm.num_groups
        x = x * valid_mask.unsqueeze(1)
        grouped = x.reshape(batch, groups, channels // groups, length)
        count = (
            valid_mask.sum(dim=-1, keepdim=True)[:, None, :, None]
            * (channels // groups)
        ).to(dtype=x.dtype)
        mean = grouped.sum(dim=(2, 3), keepdim=True) / count
        second_moment = grouped.square().sum(dim=(2, 3), keepdim=True) / count
        variance = (second_moment - mean.square()).clamp_min(0)
        normalized = ((grouped - mean) * torch.rsqrt(variance + norm.eps)).reshape_as(x)
        return normalized * norm.weight[None, :, None] + norm.bias[None, :, None]

    def forward(self, embedding, pad_mask=None):
        x = embedding.transpose(1, 2)
        valid_mask = None
        if pad_mask is not None:
            valid_mask = ~pad_mask
            x = x * valid_mask.unsqueeze(1)
        for layer in self.encoder:
            residual = x
            x = layer[0](x)
            x = self._masked_group_norm(x, layer[1], valid_mask)
            x = layer[2](x)
            x = layer[3](x) + residual
            if valid_mask is not None:
                x = x * valid_mask.unsqueeze(1)
        return x.transpose(1, 2) / len(self.encoder)


class CTBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.cnn_block = CNNBlock(d_model)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, pad_mask, return_attention=False):
        x = self.cnn_block(x, pad_mask)
        layer = self.transformer_encoder
        encoded = layer(x, src_key_padding_mask=pad_mask)
        if not return_attention:
            return encoded, None

        canonical_mask = F._canonical_mask(
            mask=pad_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(None),
            other_name="src_mask",
            target_type=x.dtype,
        )
        attention_input = layer.norm1(x) if layer.norm_first else x
        _, attention = layer.self_attn(
            attention_input,
            attention_input,
            attention_input,
            key_padding_mask=canonical_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        return encoded, attention


class Encoder(nn.Module):
    def __init__(self, layer_num, d_model, nhead, dim_feedforward=512):
        super().__init__()
        self.ct_block = nn.ModuleList(
            [
                CTBlock(d_model, nhead, dim_feedforward)
                for _ in range(layer_num)
            ]
        )

    def forward(self, node_embedding, position_embedding, pad_mask, return_attention=False):
        x = node_embedding + position_embedding
        embedding_sum = torch.zeros_like(x)
        attention_maps = []

        for layer in self.ct_block:
            x, attention = layer(x, pad_mask, return_attention=return_attention)
            embedding_sum = embedding_sum + x
            if attention is not None:
                attention_maps.append(attention)

        encoded = embedding_sum / len(self.ct_block)
        if not return_attention:
            return encoded, None
        return encoded, torch.cat(attention_maps, dim=1)


class PredictorSS(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.pair_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )
        self.pair_out = nn.Linear(embedding_dim, 1)

    def encode(self, pair_embedding):
        return self.pair_encoder(pair_embedding)

    def encode_dense(self, node_embedding):
        """Apply pair_encoder without materializing [B,L,L,2D] outer-concat."""
        first_linear = self.pair_encoder[0]
        embedding_dim = node_embedding.size(-1)
        left = F.linear(
            node_embedding,
            first_linear.weight[:, :embedding_dim],
            first_linear.bias,
        )
        right = F.linear(
            node_embedding,
            first_linear.weight[:, embedding_dim:],
            None,
        )
        pair_embedding = self.pair_encoder[1](
            left.unsqueeze(2) + right.unsqueeze(1)
        )
        pair_embedding = self.pair_encoder[2](pair_embedding)
        return self.pair_encoder[3](pair_embedding)

    def forward(self, pair_embedding):
        return self.pair_out(self.encode(pair_embedding))


class FineTuneNet(nn.Module):
    """CaTFold with its released symmetric logit-fusion adapter."""

    def __init__(
        self,
        embedding_dim,
        layer_num,
        nhead,
        adapter_hidden_dim=32,
        adapter_dropout=0.1,
        train_pair_out=False,
    ):
        super().__init__()
        self.train_pair_out = train_pair_out
        self.pretrained_frozen = False
        self._frozen_pretrained_modules = ()
        self.node_embedding = nn.Linear(4, embedding_dim)
        self.position_embedding = nn.Linear(embedding_dim, embedding_dim)
        self.encoder = Encoder(layer_num, embedding_dim, nhead, dim_feedforward=embedding_dim)
        self.predictor_ss = PredictorSS(embedding_dim)
        self.adapter = CaTFoldAdapter(
            pair_dim=embedding_dim,
            hidden_dim=adapter_hidden_dim,
            attention_channels=layer_num * nhead,
            dropout=adapter_dropout,
        )
        self.predictor_ss.pair_out.requires_grad_(train_pair_out)

    def freeze_pretrained(self):
        """Freeze the complete pretrained feature extractor and pair encoder."""
        self.pretrained_frozen = True
        self._frozen_pretrained_modules = self._pretrained_modules()
        for module in self._frozen_pretrained_modules:
            module.requires_grad_(False)
            module.eval()

    def partially_unfreeze_pretrained(self, last_n_encoder_blocks):
        """Train the pair encoder and the last N encoder blocks only."""
        blocks = self.encoder.ct_block
        if not 0 <= last_n_encoder_blocks <= len(blocks):
            raise ValueError(
                "last_n_encoder_blocks must be between 0 and "
                f"{len(blocks)}, got {last_n_encoder_blocks}"
            )

        for module in self._pretrained_modules():
            module.requires_grad_(False)

        trainable_blocks = list(blocks[-last_n_encoder_blocks:]) if last_n_encoder_blocks else []
        for module in trainable_blocks:
            module.requires_grad_(True)
        self.predictor_ss.pair_encoder.requires_grad_(True)
        self.predictor_ss.pair_out.requires_grad_(self.train_pair_out)

        frozen_blocks = list(blocks[:-last_n_encoder_blocks]) if last_n_encoder_blocks else list(blocks)
        self._frozen_pretrained_modules = (
            self.node_embedding,
            self.position_embedding,
            *frozen_blocks,
        )
        if not self.train_pair_out:
            self._frozen_pretrained_modules += (self.predictor_ss.pair_out,)
        self.pretrained_frozen = True
        for module in self._frozen_pretrained_modules:
            module.eval()

    def _pretrained_modules(self):
        return (
            self.node_embedding,
            self.position_embedding,
            self.encoder,
            self.predictor_ss,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.pretrained_frozen:
            for module in self._frozen_pretrained_modules:
                module.eval()
        return self

    def _gather_adapter_logits(self, logits, pred_pairs, seq_lengths):
        values = []
        node_offset = 0
        for batch_index, length_tensor in enumerate(seq_lengths):
            length = int(length_tensor.item())
            belongs_to_sample = (
                (pred_pairs[0] >= node_offset)
                & (pred_pairs[0] < node_offset + length)
            )
            local_pairs = pred_pairs[:, belongs_to_sample] - node_offset
            values.append(logits[batch_index, local_pairs[0], local_pairs[1]])
            node_offset += length
        return torch.cat(values)

    def forward(self, inputs):
        node_onehot, node_pe, pred_pairs, pad_mask, seq_lengths = inputs
        node_embedding = self.node_embedding(node_onehot)
        position_embedding = self.position_embedding(node_pe)
        # Evaluation uses batch_size=1 and therefore has no padding. Keep that
        # common path on the fused native GroupNorm implementation.
        encoder_pad_mask = None if pad_mask.size(0) == 1 else pad_mask
        encoded, attention_maps = self.encoder(
            node_embedding,
            position_embedding,
            encoder_pad_mask,
            return_attention=True,
        )

        pair_features = self.predictor_ss.encode_dense(encoded)
        pair_mask = None
        if encoder_pad_mask is not None:
            valid_nodes = ~encoder_pad_mask
            pair_mask = valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1)
        base_logits = self.predictor_ss.pair_out(pair_features).squeeze(-1)
        base_logits = (base_logits + base_logits.transpose(1, 2)) / 2
        adapter_matrix = self.adapter(
            pair_features,
            attention_maps,
            base_logits,
            pair_mask,
        )
        return self._gather_adapter_logits(adapter_matrix, pred_pairs, seq_lengths)

    def inference(self, inputs):
        return self.forward(inputs)


class PretrainCNNBlock(nn.Module):
    """Original pretraining CNN behavior, retained for numerical compatibility."""

    def __init__(self, embedding_dim, layer_num=3, drop_rate=0.1):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=embedding_dim),
                    nn.GELU(),
                    nn.Dropout(drop_rate),
                )
                for _ in range(layer_num)
            ]
        )

    def forward(self, embedding):
        x = embedding.transpose(1, 2)
        for layer in self.encoder:
            x = layer(x) + x
        return x.transpose(1, 2) / len(self.encoder)


class PretrainCTBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.cnn_block = PretrainCNNBlock(d_model)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, pad_mask):
        x = self.cnn_block(x)
        return self.transformer_encoder(x, src_key_padding_mask=pad_mask)


class PretrainEncoder(nn.Module):
    def __init__(self, layer_num, d_model, nhead, dim_feedforward=512):
        super().__init__()
        self.ct_block = nn.ModuleList(
            [PretrainCTBlock(d_model, nhead, dim_feedforward) for _ in range(layer_num)]
        )

    def forward(self, node_embedding, position_embedding, pad_mask):
        x = node_embedding + position_embedding
        embedding_sum = torch.zeros_like(x)
        for layer in self.ct_block:
            x = layer(x, pad_mask)
            embedding_sum = embedding_sum + x
        return embedding_sum / len(self.ct_block)


class PretrainNet(nn.Module):
    def __init__(self, embedding_dim, layer_num, nhead):
        super().__init__()
        self.node_embedding = nn.Linear(4, embedding_dim)
        self.position_embedding = nn.Linear(embedding_dim, embedding_dim)
        self.encoder = PretrainEncoder(
            layer_num, embedding_dim, nhead, dim_feedforward=embedding_dim
        )
        self.predictor_ss = PredictorSS(embedding_dim)

    def forward(self, inputs):
        node_onehot, node_pe, pred_pairs, pad_mask = inputs
        nodes = self.node_embedding(node_onehot)
        positions = self.position_embedding(node_pe)
        encoded = self.encoder(nodes, positions, pad_mask)[~pad_mask]
        emb_i = encoded[pred_pairs[0]]
        emb_j = encoded[pred_pairs[1]]
        return (
            self.predictor_ss(torch.cat((emb_i, emb_j), dim=1))
            + self.predictor_ss(torch.cat((emb_j, emb_i), dim=1))
        ).view(-1) / 2

    def inference(self, inputs):
        return self.forward(inputs)
