from typing import Union, Optional

import torch.nn
import torchvision
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop, RandomRotation, RandomPerspective

from utils.image_transforms import RandomlyAppliedTransform


class MnistDataModule(LightningDataModule):
    def __init__(
            self, root: str = '/voyager/datasets/',
            batch_size: int = 128,
            shift_transform_type: Optional[str] = None,
            test_transform_rate: Union[int, float] = 0,
            rotation: Union[int, float, tuple] = 0,
            crop: Union[int, float, tuple] = 0,
            distortion: Union[int, float] = 0,
            fashion_mnist: bool = False,
            train_val_seed: int = 42,
            test_seed: int = 42,
            test_sample_fraction: Union[int, float] = 1
    ):
        super().__init__()
        if fashion_mnist:
            self.dataset_cls = torchvision.datasets.FashionMNIST
        else:
            self.dataset_cls = torchvision.datasets.MNIST
        self.batch_size = batch_size
        self.root = root
        self.p = test_transform_rate
        self.rotation = rotation
        self.crop = crop
        self.distortion = distortion
        self.shift_transform_dict = {
            'natural': self.natural_transform(),
            'none': None,
            None: None
        }
        self.shift_transform_type = shift_transform_type
        if shift_transform_type not in self.shift_transform_dict:
            raise ValueError(f'Error: Argument {shift_transform_type} is not one of {self.shift_transform_dict.keys()}')
        self.shift_transform = self.shift_transform_dict[shift_transform_type]

        self.train_val_seed = train_val_seed
        assert 0 < test_sample_fraction <= 1, f'Expected test_sample_fraction in (0,1] got {test_sample_fraction=}'
        self.test_samples = int(10000 * test_sample_fraction)
        self.test_seed = test_seed

    def __dataloader(self, split='train') -> DataLoader:

        if split == 'train':
            transform = self.train_transforms()
        elif split == 'val':
            transform = self.val_transforms()
        else:
            raise ValueError(f'Expected split = train or val, got {split=}')
        dataset = self.dataset_cls(
            root=self.root,
            train=True,
            transform=transform,
        )
        train, val = torch.utils.data.random_split(
            dataset, [50000, 10000],
            torch.Generator().manual_seed(self.train_val_seed)
        )
        return DataLoader(
            train if split == 'train' else val,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=64,
            shuffle=True
        )

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader(split='train')

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader(split='val')

    def test_dataloader(self) -> DataLoader:
        dataset = self.dataset_cls(
            root=self.root,
            train=False,
            transform=self.test_transforms()
        )
        if (n := self.test_samples) < 10000:
            dataset = torch.utils.data.random_split(
                dataset, [n, 10000 - n],
                torch.Generator().manual_seed(self.test_seed)
            )[0]
        # make the transformation deterministic for the test dataloader
        seed_everything(self.test_seed, workers=True)
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=0
        )

    def test_transforms(self):
        # randomly apply the shifting transform, then apply the standard transform
        return Compose([self.shift_transform,
                        ToTensor(), Normalize((0.1307,), (0.3081,))
                        ])

    def natural_transform(self):
        return RandomlyAppliedTransform(transform=Compose([
            RandomRotation(self.rotation),
            RandomPerspective(distortion_scale=self.distortion, p=1),
            RandomResizedCrop(size=28, scale=self.crop if isinstance(self.crop, tuple) else (self.crop, 1.0))
        ]), p=self.p)

    def train_transforms(self):
        return Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    def val_transforms(self):
        return self.train_transforms()
