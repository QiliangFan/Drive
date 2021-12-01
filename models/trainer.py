import os
from typing import Tuple, cast
from pytorch_lightning import LightningModule
from torch import optim
import torch
from .unet import UNet
from metric.dice import Dice, DiceLoss
import imageio

class Net(LightningModule):

    def __init__(self, save_path: str = None):
        super().__init__()
        self.save_path = save_path

        self.seg_net = UNet(1)

        self.dice_loss = DiceLoss()
        self.dice = Dice()

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
        sgdr = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 4, 2, 210)
        return {
            "optimizer": opt,
            "lr_scheduler": sgdr 
        } 

    def forward(self, x):
        return self.seg_net(x)

    def training_step(self, batch, batch_idx):
        self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx)

    def step(self, batch, batch_idx):
        if isinstance(batch, Tuple):
            arr, label = batch
            out = self(arr)
            self.save_output(out, batch_idx)
            loss = self.dice_loss(out, label)
            dice = self.dice(out, label)
            self.log_dict({
                "dice": dice
            }, prog_bar=True)
            return loss
        else:
            arr = batch
            out = self(arr)
            self.save_output(out, batch_idx)
            return out

    @torch.no_grad()
    def save_output(self, out: torch.Tensor, batch_idx: int):
        if self.save_path is None:
            return

        for i, image in enumerate(out):
            image = cast(torch.Tensor, image)
            image = image.squeeze().cpu()
            imageio.imsave(os.path.join(self.save_path, f"{batch_idx + i}.png"), image)
            