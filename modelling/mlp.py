from typing import List, Tuple

import pytorch_lightning
import torch
import torch.nn as nn
from .classifier import Classifier


class ImageMLP(Classifier):
    """
    2 layer mlp + linear classifier for images
    """

    def __init__(self, in_channels=3, out_features=10, downsample_size=28, lr=1e-3):
        super(ImageMLP, self).__init__(lr)
        self.pool = nn.AdaptiveAvgPool2d((downsample_size, downsample_size))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(downsample_size * downsample_size * in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.pool(x)
        x = self.mlp(x)
        x = self.fc(x)
        return x


class ImageLinear(Classifier):
    """
    Linear classifier for images
    """

    def __init__(self, in_channels=3, out_features=10, downsample_size=28, lr=1e-3):
        super(ImageLinear, self).__init__(lr)
        self.pool = nn.AdaptiveAvgPool2d((downsample_size, downsample_size))
        self.fc = nn.Linear(downsample_size * downsample_size * in_channels, out_features)

    def forward(self, x):
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class MLP(Classifier):
    def __init__(
            self, in_features: int = 2,
            hidden_layers: Tuple[int, ...] = (8, 16, 32),
            out_features: int = 2,
            lr: float = 0.01,
            **kwargs
    ):
        super().__init__(lr=lr, **kwargs)
        self.save_hyperparameters()
        layers = [nn.Linear(in_features, hidden_layers[0]), nn.ReLU()]
        for i, j in zip(hidden_layers[:-1], hidden_layers[1:]):
            layers.extend([nn.Linear(i, j), nn.ReLU()])
        layers.append(nn.Linear(hidden_layers[-1], out_features))
        self.model = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)
