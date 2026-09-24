"""FamilyFold fusion head used by the released epoch-5 checkpoints."""

import torch
from torch import nn

from .adapters import mask_pair_grid
from .model import FineTuneNet


class ResNet2DBlock(nn.Module):
    """Bottleneck residual block from the RNA-LLM comparison protocol."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.conv_net(inputs) + inputs


class ResNet2D(nn.Module):
    def __init__(self, channels, num_blocks=2, kernel_size=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            ResNet2DBlock(channels, kernel_size) for _ in range(num_blocks)
        )

    def forward(self, inputs):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs)
        return outputs


class FamilyFoldFusionAdapter(nn.Module):
    """Fuse frozen CaTFold pair features, attention maps, and initial scores."""

    def __init__(
        self,
        pair_dim,
        hidden_dim,
        attention_channels,
        num_blocks=2,
        kernel_size=3,
    ):
        super().__init__()
        self.pair_projection = nn.Linear(pair_dim, hidden_dim)
        self.input_projection = nn.Conv2d(
            hidden_dim + attention_channels + 1,
            hidden_dim,
            kernel_size=1,
        )
        self.resnet = ResNet2D(hidden_dim, num_blocks, kernel_size)
        self.conv_out = nn.Conv2d(
            hidden_dim,
            1,
            kernel_size=kernel_size,
            padding="same",
        )

    def forward(self, pair_features, attention_maps, base_logits, pair_mask=None):
        pair_features = (pair_features + pair_features.transpose(1, 2)) / 2
        attention_maps = (attention_maps + attention_maps.transpose(2, 3)) / 2
        if pair_mask is not None:
            pair_features = pair_features * pair_mask.unsqueeze(-1)
            attention_maps = mask_pair_grid(attention_maps, pair_mask)
            base_logits = base_logits * pair_mask

        projected_pairs = self.pair_projection(pair_features).permute(0, 3, 1, 2)
        fused = torch.cat(
            (projected_pairs, attention_maps, base_logits.unsqueeze(1)), dim=1
        )
        fused = mask_pair_grid(self.input_projection(fused), pair_mask)
        logits = self.conv_out(self.resnet(fused)).squeeze(1)
        upper = torch.triu(logits, diagonal=1)
        return upper + upper.transpose(-1, -2)


class FamilyFoldFusionNet(FineTuneNet):
    """Frozen unfiltered CaTFold with the released FamilyFold fusion head."""

    def __init__(
        self,
        embedding_dim,
        layer_num,
        nhead,
        hidden_dim=64,
        num_blocks=2,
        kernel_size=3,
    ):
        super().__init__(
            embedding_dim,
            layer_num,
            nhead,
            adapter_hidden_dim=hidden_dim,
            adapter_dropout=0.0,
            train_pair_out=False,
        )
        self.adapter = FamilyFoldFusionAdapter(
            pair_dim=embedding_dim,
            hidden_dim=hidden_dim,
            attention_channels=layer_num * nhead,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
        )
