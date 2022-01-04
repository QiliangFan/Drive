from pytorch_lightning import loggers
from data.data_module import DriveDataModule
from data.utils import DriveData
import yaml
from models.trainer import Net
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning import seed_everything
import torch

def main():
    # 30-0.872
    # 5-差
    # 60-0.86
    seed_everything(30, workers=True)
    data_module = DriveDataModule(config["data"]["root"])
    net = Net(config["data"]["output"], visdom=False)

    trainer = Trainer(
        gpus=1, 
        logger=CSVLogger(save_dir="logs", name="drive"),
        callbacks=[ModelCheckpoint(dirpath="ckpt", save_weights_only=True, monitor="val_dice", mode="max")],
        benchmark=True,
        deterministic=True,
        max_epochs=210,
        # gradient_clip_val=0.1,
        check_val_every_n_epoch=10,
        # log_every_n_steps=1,
        # deterministic=True
    )
    ckpt_path = "ckpt/epoch=29-step=539.ckpt"

    # trainer.fit(net, datamodule=data_module)
    net.load_state_dict(torch.load(ckpt_path)["state_dict"])
    trainer.test(net, datamodule=data_module)


if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "rb"), yaml.FullLoader)

    main()