import torch
from torch import nn
from torch.nn.modules.activation import ELU
from torch.nn.modules.normalization import GroupNorm


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.norm = GroupNorm(8 if out_channel % 8 == 0 else out_channel, out_channel)
        self.act = ELU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))