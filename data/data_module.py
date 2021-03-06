from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from .utils import DriveData
from typing import Optional

class DriveDataModule(LightningDataModule):

    def __init__(self, data_root: str):
        drive_data = DriveData(data_root)
        
        train_data = drive_data.get_train()
        test_data = drive_data.get_test()

        self.train_data = DataLoader(train_data, batch_size=1, pin_memory=True, num_workers=4, prefetch_factor=4, shuffle=True)
        self.test_data = DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=4)

    def setup(self, stage: Optional[str] = None) -> None:
        print(f"stage: {stage}")

    def prepare_data(self) -> None:
        return super().prepare_data()

    def train_dataloader(self) -> DataLoader:
        return self.train_data

    def val_dataloader(self) -> DataLoader:
        return self.test_data

    def test_dataloader(self) -> DataLoader:
        return self.test_data