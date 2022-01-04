import os
from typing import List, Optional, Sequence, Tuple, cast
from pytorch_lightning import LightningModule
from torch import optim
import torch
from visdom import Visdom

from models.vnet import VNet
from .unet import UNet
from torch import nn 
from metric.dice import Dice, DiceLoss
from metric.clDice import soft_dice_cldice
import imageio
from .resnetx import ResNextBlock
from .block import BasicBlock, ConvBlock
from .res2net import Res2NetBlock

class Net(LightningModule):

    def __init__(self, save_path: str = None, visdom=False):
        super().__init__()
        self.save_path = save_path
        self.visdom = visdom
        if self.visdom:
            self.vis = Visdom(port=8889)
        else:
            self.vis = None
        if self.save_path is not None and not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        # self.seg_net = UNet(3)
        self.seg_net = VNet(BasicBlock)

        self.dice_loss = DiceLoss()
        # self.clDice_loss = soft_dice_cldice(alpha=0.5, iter_=3)
        # self.ce_loss = nn.BCEWithLogitsLoss()

        self.lr = 1e-2

        self.dice = Dice()

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False
    ):
        if self.trainer.global_step < 100:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 100)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        # sgdr = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 4, 2)
        sgdr = optim.lr_scheduler.CosineAnnealingLR(opt, 10)
        return {
            "optimizer": opt,
            "lr_scheduler": sgdr 
        } 

    def forward(self, x):
        return self.seg_net(x)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, save=False)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, save=False)

    def step(self, batch, batch_idx, save=True):
        cur_epoch = self.trainer.current_epoch
        if isinstance(batch, Sequence) and len(batch) == 3:
            arr, mask, label = batch
            arr = arr * mask
            out = self(arr)
            out = out * mask
            if self.vis and not self.training:
                for i, (_out, _arr, _label) in enumerate(zip(out, arr, label)):
                    self.vis.image((_out.squeeze() > 0.5).type(torch.float32), win=f"{(batch_idx + i) % 5}_out")
                    self.vis.image(_arr.squeeze(), win=f"{(batch_idx + i) % 5}_arr")
                    self.vis.image(_label.squeeze(), win=f"{(batch_idx + i) % 5}_label")

            if save:
                self.save_output(out, batch_idx)

            loss = self.dice_loss(out, label)
            # cl_loss = self.clDice_loss(out, label)  # 比 ce 还差
            # loss = self.ce_loss(out, label)   # 很差
            # alpha = 1 - (cur_epoch // 5) * 0.05
            # loss = alpha * dice_loss + (1 - alpha) * cl_loss

            dice = self.dice(out, label)
            self.log_dict({
                f"{'train' if self.training else 'val'}_dice": dice,
                # "lr": self.optimizers().state_dict()["param_groups"][0]['lr']
            }, prog_bar=True)
            return loss
        else:
            arr, mask = batch
            out = self(arr)
            self.save_output(out, batch_idx)
            return out

    @torch.no_grad()
    def save_output(self, out: torch.Tensor, batch_idx: int):
        if self.save_path is None and self.training:
            return

        for i, image in enumerate(out):
            image = cast(torch.Tensor, image)
            image = image.squeeze().cpu()
            image = image * 255
            image = image.type(torch.uint8)
            imageio.imsave(os.path.join(self.save_path, f"{batch_idx + i}.png"), image)
            