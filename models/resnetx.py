import torch
from torch import nn

def conv_block(in_channel, channel, kernel_size=3, stride=1, padding=1, inplace=False):
    norm = nn.InstanceNorm2d
    return nn.Sequential(
        nn.Conv2d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        norm(channel, affine=True, momentum=0.4),
        nn.ELU(inplace=inplace)
    )



class ResNextBlock(nn.Module):
    """
    resnext不是太好，将原先的+改为了cat，残差不变
    我想命名为 ResNextX
    """

    def __init__(self, in_channel, out_channel, split_num: int = 4, **kwargs):
        super().__init__()

        mid_channel = 4

        self.split_num = split_num
        blocks = []

        kernel_style = [
            (1, 1),
            (1, 3),
            (3, 1),
            (3, 3)
        ]

        padding_style = [
            (k1 // 2, k2 // 2) for k1, k2 in kernel_style
        ]
        inplace = kwargs.get("inplace", True)
        for i in range(split_num):
            blocks.append(nn.Sequential(
                conv_block(in_channel, mid_channel, kernel_size=kernel_style[i], stride=1, padding=padding_style[i], inplace=inplace),
                conv_block(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, inplace=inplace),
                conv_block(mid_channel, out_channel, kernel_size=kernel_style[i], stride=1, padding=padding_style[i], inplace=inplace),
                # conv_block(out_channel, mid_channel, kernel_size=1, stride=1, padding=0, inplace=inplace),
                # conv_block(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, inplace=inplace),
                # conv_block(mid_channel, out_channel, kernel_size=1, stride=1, padding=0, inplace=inplace)
            ))
        self.scale = conv_block(out_channel * split_num, out_channel, kernel_size=1, stride=1, padding=0, inplace=inplace)
        self.blocks = nn.ModuleList(blocks)
        
        if in_channel != out_channel:
            self.skip = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        else:
            self.skip = None


    def forward(self, x):
        if self.skip:
            res = self.skip(x)
        else:
            res = x

        outputs = []
        for i in range(self.split_num):
            outputs.append(torch.add(self.blocks[i](x), res))

        x = self.scale(torch.cat(outputs, dim=1))

        
        return torch.add(x, res)
        