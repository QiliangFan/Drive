from typing import List, Optional
from matplotlib.pyplot import scatter
import torch
from torch import nn
from torch.nn.modules.instancenorm import InstanceNorm2d


class FirstBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, conv_block):
        super().__init__()

        self.conv1 = conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=1, inplace=True)

        self.conv2 = conv_block(out_channel, out_channel, kernel_size=3, stride=1, padding=1, inplace=True)

        if in_channel != out_channel:
            self.skip = nn.Conv2d(in_channel, out_channel, kernel_size=1,
                                  stride=1, padding=0)
        else:
            self.skip = None

    def forward(self, x):
        if self.skip:
            res = self.skip(x)
        else:
            res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.add(x, res)
        return x


class DownBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, conv_block):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(in_channel, momentum=1, affine=True),
            nn.ELU(inplace=True)
        )

        self.conv1 = conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=1, inplace=True)

        self.conv2 = conv_block(out_channel, out_channel, kernel_size=3, stride=1, padding=1, inplace=True)

        if in_channel != out_channel:
            self.skip = nn.Conv2d(in_channel, out_channel, kernel_size=1,
                                  stride=1, padding=0)
        else:
            self.skip = None

    def forward(self, x):
        x = self.down_sample(x)
        if self.skip:
            res = self.skip(x)
        else:
            res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.add(x, res)
        return x


class LastBlock(nn.Module):

    def __init__(self, in_channel: int, conv_block):
        super().__init__()

        self.conv1 = conv_block(in_channel, in_channel, kernel_size=3, stride=1, padding=1, inplace=True)
        self.conv2 = conv_block(in_channel, in_channel, kernel_size=3, stride=1, padding=1, inplace=True)
        self.conv3 = conv_block(in_channel, in_channel, kernel_size=3, stride=1, padding=1, inplace=True)
        self.act = nn.Sigmoid()
        # self.skip = nn.Conv2d(in_channel, 1, kernel_size=1,
        #                       stride=1, padding=0, groups=groups2(in_channel, 1))

        self.last = nn.Conv2d(in_channel, 1, 1, 1, 0)

    def forward(self, x):
        # res = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = torch.add(x, res)
        x = self.last(x)
        x = self.act(x)
        return x


class UpBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, conv_block):
        super().__init__()

        self.conv1 = conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=1, inplace=True)
        self.conv2 = conv_block(out_channel, out_channel, kernel_size=3, stride=1, padding=1, inplace=True)

        if in_channel != out_channel:
            self.skip = nn.Conv2d(in_channel, out_channel, 1,
                                  1, 0)
        else:
            self.skip = None

    def forward(self, x):
        if self.skip:
            res = self.skip(x)
        else:
            res = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.add(x, res)
        return x


class DownLayer(nn.Module):

    def __init__(self, num_pooling: int,  conv_block):
        super().__init__()
        self.num_pooling = num_pooling

        cur_channel = 3
        next_channel = 16
        self.expand = 4

        blocks = []
        blocks.append(FirstBlock(cur_channel, next_channel, conv_block))
        cur_channel, next_channel = next_channel, next_channel * self.expand

        for i in range(num_pooling):
            blocks.append(DownBlock(cur_channel, next_channel, conv_block))
            if i == num_pooling - 2:
                cur_channel, next_channel = next_channel, next_channel
            else:
                cur_channel, next_channel = next_channel, next_channel * self.expand

        self.out_channel = cur_channel
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outputs = []
        for i in range(self.num_pooling+1):
            x = self.blocks[i](x)
            outputs.append(x)
        return list(reversed(outputs))


class UpLayer(nn.Module):

    def __init__(self, num_pooling: int, in_channel: int, expand: int, conv_block):
        super().__init__()

        self.num_pooling = num_pooling

        cur_channel = in_channel
        next_channel = in_channel // expand

        blocks = []
        up_samples = []
        for i in range(num_pooling - 1):
            # up_samples.append(nn.ConvTranspose2d(cur_channel, cur_channel, kernel_size=2, stride=2, padding=0, output_padding=0))
            up_samples.append(nn.Sequential(
                nn.ConvTranspose2d(cur_channel, cur_channel, kernel_size=2, stride=2, padding=0, output_padding=0),
                nn.InstanceNorm2d(cur_channel, momentum=0.4, affine=True),
                nn.ELU(inplace=True)
            ))   
            blocks.append(UpBlock(cur_channel * 2, next_channel, conv_block))
            cur_channel, next_channel = next_channel, next_channel // expand
        # up_samples.append(nn.ConvTranspose2d(cur_channel, cur_channel, kernel_size=2, stride=2, padding=0, output_padding=0))
        up_samples.append(nn.Sequential(
            nn.ConvTranspose2d(cur_channel, cur_channel, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.InstanceNorm2d(cur_channel, affine=True, momentum=0.4),
            nn.ELU(inplace=True)
        ))
        blocks.append(LastBlock(cur_channel * 2, conv_block))
        self.blocks = nn.ModuleList(blocks)
        self.up_samples = nn.ModuleList(up_samples)

    def forward(self, inputs: List[torch.Tensor]):
        assert self.num_pooling > 0, self.num_pooling
        last: Optional[torch.Tensor] = None
        for i in range(self.num_pooling+1):
            if i == 0:
                last = inputs[i]
            else:
                last = self.blocks[i -1](torch.cat([self.up_samples[i - 1](last), inputs[i]], dim=1))
        return last


class VNet(nn.Module):

    def __init__(self, conv_block, num_pooling=4):
        super().__init__()

        self.num_pooling = 8

        self.down_layer = DownLayer(num_pooling, conv_block)

        self.up_layer = UpLayer(
            num_pooling, self.down_layer.out_channel, self.down_layer.expand, conv_block)

    def forward(self, x):
        outputs = self.down_layer(x)
        return self.up_layer(outputs)
