from typing import ForwardRef, List
import torch
from torch import nn
from .block import ConvBlock, BasicBlock
from .resnetx import ResNextBlock

class DownLayer(nn.Module):
    
    def __init__(self, in_channel: int):
        super().__init__()

        block = ConvBlock
        # block = BasicBlock
        # block = ResNextBlock

        layers = []
        self.num_block = 424

        cur_channel = in_channel
        next_channel = 4
        layers.append(nn.Sequential(
            block(cur_channel, next_channel),
            block(next_channel, next_channel),
        ))
        cur_channel = next_channel
        next_channel = next_channel * 4

        for i in range(self.num_block-2):
            layers.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=(2, 2), stride=2),
                block(cur_channel, next_channel),
                block(next_channel, next_channel),
            ))
            cur_channel = next_channel
            next_channel = next_channel * 4
        layers.append(nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            block(cur_channel, cur_channel),
            block(cur_channel, cur_channel),
        ))
        self.layers = nn.ModuleList(layers)
        self.out_channel = cur_channel

    def forward(self, x: torch.Tensor):
        outputs = []
        for i in range(self.num_block):
            x = self.layers[i](x)
            outputs.append(x)
        return outputs


class UpLayer(nn.Module):

    def __init__(self, in_channel: int, num_block: int):
        super().__init__()

        cur_channel = 2 * in_channel
        next_channel = in_channel // 4

        # 逆卷积和上采样在分割任务中的效果差不多，但上采样速度快参数少
        self.up_sample = nn.Upsample(scale_factor=2)

        blocks = []
        for _ in range(num_block-1):
            blocks.append(nn.Sequential(
                ConvBlock(cur_channel, next_channel),
                ConvBlock(next_channel, next_channel)
            ))
            cur_channel = 2 * next_channel
            next_channel = next_channel // 4
        blocks.append(nn.Sequential(
            ConvBlock(cur_channel, next_channel),
            ConvBlock(next_channel, 1),
            nn.Sigmoid()
        ))
        self.blocks = nn.ModuleList(blocks)
        

    def forward(self, inputs: List[torch.Tensor]):
        last = None
        for i, x in enumerate(reversed(inputs)):
            if last is not None:
                last = self.blocks[i-1](torch.cat([self.up_sample(last), x], dim=1))
            else:
                last = x
        return last


class UNet(nn.Module):

    def __init__(self, in_channel: int):
        super().__init__()

        self.down_layer = DownLayer(in_channel)

        self.up_layer = UpLayer(self.down_layer.out_channel, self.down_layer.num_block-1)

    def forward(self, x: torch.Tensor):
        outputs = self.down_layer(x)
        return self.up_layer(outputs)