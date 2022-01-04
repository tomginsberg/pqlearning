import random
from os.path import join
from typing import Union, Collection, Tuple

import numpy as np
import torch.nn
import torchvision
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets.flipped import FlippedLabels

TransformType = Union[
    torchvision.transforms.Compose, torch.nn.Module, torchvision.transforms.ToTensor]


class CIFAR10C(Dataset):
    def __init__(self, root='/voyager/datasets/', shift_types=('frost', 'fog', 'snow'), seed=42, num_images=10000,
                 severity_range=(3, 5), negative_label=True):
        self.num_images = num_images
        _images = [np.load(join(root, 'CIFAR-10-C', x + '.npy')) for x in shift_types]
        _labels = np.load(join(root, 'CIFAR-10-C', 'labels.npy')).astype(int)

        rd = random.Random(seed)
        severity = np.array([rd.randint(severity_range[0] - 1, severity_range[1] - 1) for _ in range(num_images)])
        corruption = np.array([rd.randint(0, len(shift_types) - 1) for _ in range(num_images)])
        subset = np.array(rd.sample(range(10000), num_images))
        images = []
        labels = []
        for s, c, idx in zip(severity, corruption, subset):
            # severity increases by 1 every 10,000 images
            images.append(Image.fromarray(_images[c][idx + 10000 * s]))
            labels.append(_labels[idx + 10000 * s])
        del _images, _labels
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.images = images
        self.labels = labels
        self.nl = negative_label

    def __len__(self):
        return self.num_images

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        l = -self.labels[index] - 1 if self.nl else self.labels[index]
        return self.tf(self.images[index]), l


class CIFAR10DataModule(LightningDataModule):
    shift_options = {'brightness',
                     'frost',
                     'jpeg_compression',
                     'shot_noise',
                     'contrast',
                     'gaussian_blur',
                     'labels',
                     'snow',
                     'defocus_blur',
                     'gaussian_noise',
                     'motion_blur',
                     'spatter',
                     'elastic_transform',
                     'glass_blur',
                     'pixelate',
                     'speckle_noise',
                     'fog',
                     'impulse_noise',
                     'saturate',
                     'zoom_blur'}

    def __init__(
            self, root: str = '/voyager/datasets/',
            batch_size: int = 512,
            test_seed: int = 42,
            test_samples: Union[int, str] = 10000,
            shift_types: Union[str, Collection[str]] = ('frost', 'fog', 'snow'),
            shift_severity_range=(3, 5),
            unshifted_test=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.shift_type = shift_types
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(root, train=True, transform=self.transform_train)
        self.val_dataset = torchvision.datasets.CIFAR10(root, train=False, transform=self.transform_val)
        self.batch_size = batch_size
        self.root = root

        assert 1 <= test_samples <= 10000, f'Expected 10,000 >= test_samples >= 1 got {test_samples=}'
        self.test_samples = test_samples
        self.test_seed = test_seed

        if shift_types == 'all':
            shift_types = self.shift_options
        else:
            for x in shift_types:
                assert x in self.shift_options, f'shift type {x} is not in available shift types {self.shift_options}'
        if not unshifted_test:
            self.test_dataset = CIFAR10C(root=root, shift_types=shift_types, severity_range=shift_severity_range,
                                         negative_label=True, seed=test_seed, num_images=test_samples)
        else:
            self.test_dataset = torchvision.datasets.CIFAR10(root, train=False, transform=self.transform_train)
            if self.test_samples < 10000:
                self.test_dataset = torch.utils.data.random_split(
                    self.test_dataset, [self.test_samples, 10000 - self.test_samples],
                    torch.Generator().manual_seed(self.test_seed)
                )[0]
            self.test_dataset = FlippedLabels(self.test_dataset)

    def __dataloader(self, split='train') -> DataLoader:
        return DataLoader(
            self.train_dataset if split == 'train' else self.val_dataset,
            batch_size=self.batch_size,
            # pin_memory=True,
            num_workers=64,
            shuffle=True if split == 'train' else False,
            persistent_workers=True
        )

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader(split='train')

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader(split='val')

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=64,
            shuffle=True,
            persistent_workers=True

        )

    def update_test_transform(self, train=True):
        """
        :param train: bool, if true uses the train transform for the test set (i.e data augmentation)
        if not uses the default validation transform (i.e normalize + ToTensor)
        """
        if train:
            self.test_dataset.tf = self.transform_train
        else:
            self.test_dataset.tf = self.transform_val
