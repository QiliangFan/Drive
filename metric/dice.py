import torch
from torch import nn


class Dice:

    def __init__(self):
        pass

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        output = output.flatten()
        target = target.flatten()

        inter = output * target
        union = torch.pow(output, 2).sum() + torch.pow(target, 2).sum()

        return (2 * inter / (union + 1e-6)).mean()

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.dice = Dice()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return 1 - self.dice(output, target)