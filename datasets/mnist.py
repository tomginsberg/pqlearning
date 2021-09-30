import torchvision
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop, RandomRotation, RandomPerspective
from utils.image_transforms import RandomlyAppliedTransform


class MnistDataModule(LightningDataModule):
    def __init__(
            self, root='~/datasets',
            batch_size=128,
            shift_transform_type=None,
            test_transform_rate=0,
            rotation=0,
            crop=0,
            distortion=0
    ):
        super().__init__()
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

    def dataloader(self, train=True) -> DataLoader:
        if train:
            transform = self.train_transforms()
        else:
            transform = self.val_transforms()
        return DataLoader(
            torchvision.datasets.MNIST(
                root=self.root,
                train=train,
                transform=transform
            ),
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=0
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(train=True)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(train=False)

    def test_dataloader(self) -> DataLoader:
        seed_everything(42, workers=True)
        return DataLoader(
            torchvision.datasets.MNIST(
                root=self.root,
                train=False,
                transform=self.test_transforms()
            ),
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
            RandomResizedCrop(size=28, scale=(self.crop, 1.0))
        ]), p=self.p)

    def train_transforms(self):
        return Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    def val_transforms(self):
        return self.train_transforms()
