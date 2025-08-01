import torch
from torch import nn

class PairCNN(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, bias=True):
        super().__init__()

        self.cnn_block = nn.Sequential(
                            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, bias=bias, padding=kernel_size//2),
                            nn.GroupNorm(num_groups=8, num_channels=embed_dim),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            )

    def forward(self, x):
        x = self.cnn_block(x)
        return x

class CNNAdapter(nn.Module):
    def __init__(self, in_dim, conv_dim):
        super().__init__()
        
        self.linear_in = nn.Linear(in_dim, conv_dim)
        self.conv_in = nn.Conv2d(conv_dim+64, conv_dim, 1)

        self.fusion_cnn = PairCNN(conv_dim)
        self.fusion_out = nn.Conv2d(conv_dim, 1, kernel_size=1)
        
    def forward(self, input):
        pair_embedding, attn_maps = input
        x = self.linear_in(pair_embedding).permute(0, 3, 1, 2)
        x = torch.cat((x, attn_maps), dim=1)
        x = self.conv_in(x)
        
        x = self.fusion_cnn(x)
        x = self.fusion_out(x)
        
        x = (x + x.transpose(2 ,3)) / 2
        output = x.squeeze(1)

        return output