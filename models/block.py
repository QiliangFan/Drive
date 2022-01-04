import torch
from torch import nn
from torch.nn.modules.activation import ELU
from torch.nn.modules.normalization import GroupNorm


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        # self.norm = GroupNorm(8 if out_channel % 8 == 0 else out_channel, out_channel)
        self.norm = nn.InstanceNorm2d(out_channel, momentum=0.4, affine=True)
        self.act = ELU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.relu1 = ELU(inplace=True)
        self.norm1 = nn.InstanceNorm2d(out_channel, momentum=0.4, affine=True)
        # self.norm1 = nn.InstanceNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = ELU(inplace=True)
        self.norm2 = nn.InstanceNorm2d(out_channel, momentum=0.4, affine=True)
        # self.norm2 = nn.InstanceNorm2d(out_channel)

        if stride == 2 and in_channel == out_channel:
            self.skip = nn.AvgPool2d(2, 2)

        elif in_channel != out_channel:
            self.skip = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

        else:
            self.skip = None

    def forward(self, x):
        if self.skip:
            res = self.skip(x)
        else:
            res = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x += res
        x = self.norm2(x)
        x = self.relu2(x)

        return x

        