import json
from os.path import join
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class Rectangles(Dataset):
    def __init__(self, root='/voyager/projects/tomginsberg/pqlearning/synthetic_data/rectangles', split='train',
                 negative_test_labels=True):
        self.root = root
        self.pos_label = 1
        self.neg_label = 0
        if split == 'train':
            positive = self.read_data('positive')
            negative = self.read_data('negative')

        elif split == 'val':
            positive = self.read_data('val_positive')
            negative = self.read_data('val_negative')

        elif split == 'test':
            test = self.read_data('test')
            # y > sin(x)
            positive = test[test[:, 1] >= torch.sin(test[:, 0])]
            negative = test[test[:, 1] < torch.sin(test[:, 0])]
            if negative_test_labels:
                self.pos_label = -2
                self.neg_label = -1
        else:
            raise ValueError(f'{split=} not train, val or test')

        self.p = positive
        self.n = negative

    def read_data(self, name):
        return torch.tensor(json.load(open(join(self.root, name + '.json'), 'r')))

    def __len__(self):
        return 200

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if index < len(self.p):
            return self.p[index], self.pos_label
        else:
            return self.n[index - len(self.p)], self.neg_label


class RectanglesModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train = Rectangles(split='train')
        self.val = Rectangles(split='val')
        self.test = Rectangles(split='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, num_workers=10, shuffle=True, batch_size=20, pin_memory=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, num_workers=10, shuffle=True, batch_size=20, pin_memory=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, num_workers=10, shuffle=True, batch_size=20, pin_memory=False)
