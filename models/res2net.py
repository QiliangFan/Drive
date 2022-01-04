import torch
from torch import nn
from torch.nn import ELU


class Res2NetBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, inplace=True):
        super().__init__()

        if out_channel % 2 == 0:
            self.split_num = 2
        else:
            self.split_num = 1
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu1 = ELU(inplace=inplace)
        self.norm1 = nn.InstanceNorm2d(out_channel, momentum=0.4, affine=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu2 = ELU(inplace=inplace)
        self.norm2 = nn.InstanceNorm2d(out_channel, momentum=0.4, affine=True)

        self.conv_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel // self.split_num, out_channel // self.split_num, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(self.out_channel // self.split_num, momentum=0.4, affine=True),
                ELU(inplace=inplace)
            ) for _ in range(self.split_num-1)
        ])

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
        slices = torch.split(x, self.out_channel // self.split_num, dim=1)
        sp = None
        result = None
        for i in range(self.split_num):
            if result is None:
                result = slices[0]
            else:
                if sp is None:
                    sp = self.conv_list[i-1](slices[i])
                else:
                    sp = torch.add(sp, self.conv_list[i-1](slices[i]))
                result = torch.cat([result, sp], dim=1)

        x = self.conv2(result)
        x = torch.add(x, res)
        return x