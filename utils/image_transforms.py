from typing import Union

import torch
from torchvision.transforms import Compose


class RandomlyAppliedTransform(torch.nn.Module):
    def __init__(self, transform: Union[Compose, torch.nn.Module], p: float = 0.5):
        """
        Apply a transform with probability p
        :param transform: to be applied
        :param p: probability, 0<=p<=1
        """
        super().__init__()
        self.transform = transform
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return self.transform(img)
        return img
