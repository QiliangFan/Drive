from pytorch_lightning import loggers
from data.data_module import DriveDataModule
from data.utils import DriveData
import yaml
from models.trainer import Net
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning import seed_everything

def main():
    seed_everything(20, workers=True)
    data_module = DriveDataModule(config["data"]["root"])
    net = Net(config["data"]["output"], visdom=False)

    trainer = Trainer(
        gpus=1, 
        logger=CSVLogger(save_dir="logs", name="drive"),
        callbacks=[ModelCheckpoint(dirpath="ckpt", save_weights_only=True, monitor="val_dice", mode="max")],
        deterministic=True,
        max_epochs=210,
        gradient_clip_val=0.1,
        check_val_every_n_epoch=10,
        # log_every_n_steps=1,
        # deterministic=True
    )

    trainer.fit(net, datamodule=data_module)
    trainer.test(net, datamodule=data_module)


if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "rb"), yaml.FullLoader)

    main()