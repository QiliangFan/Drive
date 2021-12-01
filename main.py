from pytorch_lightning import loggers
from data.data_module import DriveDataModule
from data.utils import DriveData
import yaml
from models.trainer import Net
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

def main():
    data_module = DriveDataModule(config["data"]["root"])
    net = Net(config["data"]["output"], visdom=True)

    trainer = Trainer(
        gpus=0, 
        logger=CSVLogger(save_dir="logs", name="drive"),
        callbacks=[ModelCheckpoint(dirpath="ckpt", save_weights_only=True), ModelSummary(max_depth=4)],
        max_epochs=210,
        # val_check_interval=10,
        log_every_n_steps=1,
        # deterministic=True
    )

    trainer.fit(net, datamodule=data_module)
    trainer.test(net, datamodule=data_module)


if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "rb"), yaml.FullLoader)

    main()