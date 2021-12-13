import os
from glob import glob
from typing import Tuple
import numpy as np
import torch
import imageio
from torch.utils.data import Dataset
from torchvision import transforms

class TestItem(Dataset):

    def __init__(self, test_root: str, transform):
        super().__init__()
        assert os.path.exists(test_root), test_root
        self.transform = transform
        images = glob(os.path.join(test_root, "images"))
        masks = glob(os.path.join(test_root, "mask"))

        assert len(images) == len(masks)
        self.images = images
        self.masks = masks

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        mask = self.masks[idx]
        img = imageio.imread(img)
        img = np.einsum("hwc->chw", img)
        mask = imageio.imread(mask)
        img, mask = torch.as_tensor(img, dtype=torch.float32), \
            torch.as_tensor(mask, dtype=torch.float32).unsqueeze(dim=0)
        img, mask = (img - img.min()) / (img.max() - img.min()), \
            (mask - mask.min()) / (mask.max() - mask.min())
        mask = (mask > 0.5).type(torch.float32)
        img, mask = self.transform(img), self.transform(mask)
        img = (img - img.min()) / (img.max() - img.min())
        return img, mask

    def __len__(self):
        return len(self.images)


class TrainItem(Dataset):

    def __init__(self, train_root: str, transform):
        super().__init__()
        assert os.path.exists(train_root), train_root
        self.transform = transform
        images = glob(os.path.join(train_root, "images", "*"))
        masks = glob(os.path.join(train_root, "mask", "*"))
        labels = glob(os.path.join(train_root, "1st_manual", "*"))
        images.sort(key=lambda x: int(os.path.basename(x)[:2]))
        masks.sort(key=lambda x: int(os.path.basename(x)[:2]))
        labels.sort(key=lambda x: int(os.path.basename(x)[:2]))

        assert len(images) == len(masks) == len(labels)
        self.images = images
        self.masks = masks
        self.labels = labels

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        img = imageio.imread(img)
        img = np.einsum("hwc->chw", img)
        mask = imageio.imread(mask)
        label = imageio.imread(label)
        img, mask, label = torch.as_tensor(img, dtype=torch.float32), \
            torch.as_tensor(mask, dtype=torch.float32).unsqueeze(dim=0), \
            torch.as_tensor(label, dtype=torch.float32).unsqueeze(dim=0)
        img, mask, label = (img - img.min()) / (img.max() - img.min()), \
            (mask - mask.min()) / (mask.max() - mask.min()), \
            (label - label.min()) / (label.max() - label.min())
        mask = (mask > 0.5).type(torch.float32)
        img, mask, label = self.transform(img), self.transform(mask), self.transform(label)
        img = (img - img.min()) / (img.max() - img.min())
        return img, mask, label

    def __len__(self):
        return len(self.images)



class DriveData:
    """
    load the datasets
    """
    def __init__(self, data_root: str):
        data_root = os.path.expanduser(data_root)  # ~ -> /home/username
        self.train_root = os.path.join(data_root, "training")
        self.test_root = os.path.join(data_root, "test")

        train_transform = transforms.Compose([
            transforms.Resize([512, 512]),
        ])

        test_transform = transforms.Compose([
            transforms.Resize([512, 512]),
        ])

        self.train_data = TrainItem(train_root=self.train_root, transform=train_transform)
        self.test_data = TestItem(test_root=self.test_root, transform=test_transform)

    def get_train(self):
        return self.train_data

    def get_test(self):
        return self.test_data