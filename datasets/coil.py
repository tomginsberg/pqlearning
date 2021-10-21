import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor


class Coil100(Dataset):
    to_tensor = ToTensor()

    def __init__(self, split: str = 'p1'):
        self.df = pd.read_csv('/voyager/datasets/coil-100/annotations.csv')
        assert split in {'all', 'p1', 'p2', 'q1', 'q2'}
        if split != 'all':
            self.df = self.df[self.df.split == split]
        self.split = split
        # self.images = [Coil100.to_tensor(Image.open(x)) for x in self.df.path.iloc]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        image = Coil100.to_tensor(Image.open(self.df.iloc[index].path))
        return image, self.df.iloc[index].obj - 1


class Coil100Module(pl.LightningDataModule):
    def __init__(self, batch_size=225, split=1):
        super().__init__()
        self.batch_size = batch_size
        assert split in {1, 2}
        self.train_dataset = Coil100(split=f'p{split}')
        self.val_dataset = Coil100(split=f'p{2 if split == 1 else 1}')
        self.test_dataset = Coil100(split=f'q{split}')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, shuffle=True, batch_size=self.batch_size)
