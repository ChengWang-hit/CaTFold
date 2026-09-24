"""Checkpoint-compatible adapter used by all released fine-tuned models."""

import torch
from torch import nn


def mask_pair_grid(x, pair_mask):
    if pair_mask is None:
        return x
    return x * pair_mask.unsqueeze(1).to(dtype=x.dtype)


def masked_group_norm(x, norm, pair_mask):
    if pair_mask is None:
        return norm(x)
    batch, channels, height, width = x.shape
    groups = norm.num_groups
    x = mask_pair_grid(x, pair_mask)
    grouped = x.reshape(batch, groups, channels // groups, height, width)
    count = (
        pair_mask.sum(dim=(1, 2), keepdim=True)[:, None, :, :, None]
        * (channels // groups)
    ).to(dtype=x.dtype)
    mean = grouped.sum(dim=(2, 3, 4), keepdim=True) / count
    second_moment = grouped.square().sum(dim=(2, 3, 4), keepdim=True) / count
    variance = (second_moment - mean.square()).clamp_min(0)
    normalized = ((grouped - mean) * torch.rsqrt(variance + norm.eps)).reshape_as(x)
    normalized = (
        normalized * norm.weight[None, :, None, None]
        + norm.bias[None, :, None, None]
    )
    return mask_pair_grid(normalized, pair_mask)


class DilationThreePairCNN(nn.Module):
    """One-layer local plus dilation-3 context block."""

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        if channels % 2:
            raise ValueError("Pair channels must be even")
        branch_channels = channels // 2
        self.local = nn.Conv2d(channels, branch_channels, kernel_size=3, padding=1)
        self.dilated = nn.Conv2d(
            channels,
            branch_channels,
            kernel_size=3,
            padding=3,
            dilation=3,
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair_mask=None):
        x = torch.cat((self.local(x), self.dilated(x)), dim=1)
        x = masked_group_norm(x, self.norm, pair_mask)
        x = self.activation(x)
        return self.dropout(mask_pair_grid(x, pair_mask))


class CaTFoldAdapter(nn.Module):
    """Symmetric logit-fusion adapter used by the released model."""

    def __init__(self, pair_dim, hidden_dim, attention_channels, dropout=0.1):
        super().__init__()
        self.pair_projection = nn.Linear(pair_dim, hidden_dim)
        self.input_projection = nn.Conv2d(
            hidden_dim + attention_channels + 1,
            hidden_dim,
            kernel_size=1,
        )
        self.fusion = DilationThreePairCNN(hidden_dim, dropout=dropout)
        self.output = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, pair_features, attention_maps, base_logits, pair_mask=None):
        pair_features = (pair_features + pair_features.transpose(1, 2)) / 2
        attention_maps = (attention_maps + attention_maps.transpose(2, 3)) / 2
        if pair_mask is not None:
            pair_features = pair_features * pair_mask.unsqueeze(-1)
            attention_maps = mask_pair_grid(attention_maps, pair_mask)
            base_logits = base_logits * pair_mask
        x = self.pair_projection(pair_features).permute(0, 3, 1, 2)
        x = torch.cat((x, attention_maps, base_logits.unsqueeze(1)), dim=1)
        x = self.input_projection(x)
        x = mask_pair_grid(x, pair_mask)
        x = self.fusion(x, pair_mask)
        x = mask_pair_grid(x, pair_mask)
        logits = self.output(x).squeeze(1)
        return (logits + logits.transpose(1, 2)) / 2
