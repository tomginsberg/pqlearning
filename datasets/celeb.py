import os
from typing import Tuple, Union, Optional

import pandas as pd
import torch
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import PIL

# noinspection PyUnresolvedReferences
PIL_IMAGE = PIL.JpegImagePlugin.JpegImageFile


class CelebDataset(Dataset):
    def __init__(self, target_label: str = 'Attractive', filter_by_label: Optional[str] = 'Young',
                 take_positives: bool = True,
                 train=True, split_seed=42, take_all=False, return_pil=False):
        self.root = '/voyager/datasets/CelebA/img_align_celeba/'
        attr = '/voyager/datasets/CelebA/list_attr_celeba.txt'
        attr = pd.read_csv(attr, sep='\s+')
        if filter_by_label:
            attr = attr[attr[filter_by_label] == (1 if take_positives else -1)]

        if take_all:
            # Take all images, no train/test split
            self.attr = attr
        else:
            # if train take a random subset of the 95% data seeded by split_seed
            train_attr = attr.sample(frac=0.95, random_state=split_seed)
            if not train:
                # take the complement of attr with train
                self.attr = attr.drop(train_attr.index)
            else:
                self.attr = train_attr

        self.target_label = target_label
        self.return_pil = return_pil

    def __len__(self) -> int:
        return len(self.attr)

    def __getitem__(self, index) -> Tuple[Union[PIL_IMAGE, torch.Tensor], int]:
        row = self.attr.iloc[index]
        image = Image.open(os.path.join(self.root, row['File_Name']))
        label = row[self.target_label]
        label = 1 if label == 1 else 0

        # pad to square
        image = transforms.Pad(padding=5)(image)
        image = transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(1, 1))(image)

        if not self.return_pil:
            # normalize to [-1, 1]
            image = transforms.ToTensor()(image)

        return image, label


class CelebModule(LightningDataModule):

    def __init__(self, target_label: str = 'Attractive', filter_by_label: str = 'Young',
                 split_seed=42, num_workers=96 // 2, batch_size=512):
        super().__init__()
        self.dataset = lambda _train, _take_all=False, _take_positives=True: CelebDataset(target_label=target_label,
                                                                                          filter_by_label=filter_by_label,
                                                                                          take_positives=_take_positives,
                                                                                          train=_train,
                                                                                          split_seed=split_seed,
                                                                                          take_all=_take_all)
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset = self.dataset(_train=True)
        self.val_dataset = self.dataset(_train=False)
        self.test_dataset = self.dataset(_train=False, _take_all=True, _take_positives=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError
