from typing import Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch import nn


class CNN(pl.LightningModule):
    """
    Base class for CNN models
    Default architecture is Resnet 18
    """

    def __init__(self,
                 out_features: int = 10,
                 in_channels: int = 3,
                 negative_labels=False,
                 training_multiplier=1,
                 learning_rate=0.1,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(pretrained=False)
        if out_features != self.model.fc.out_features:
            self.model.fc = nn.Linear(self.model.fc.in_features, out_features, bias=True)
        if in_channels != 3:
            conv1 = self.model.conv1
            self.model.conv1 = nn.Conv2d(in_channels=in_channels,
                                         out_channels=conv1.out_channels,
                                         kernel_size=conv1.kernel_size,
                                         padding=conv1.padding, bias=conv1.bias
                                         )
        self.negative_labels = negative_labels
        self.training_multiplier = training_multiplier
        self.learning_rate = learning_rate

    def forward(self, x) -> Tensor:
        return self.model(x)

    def train_val_step(self, batch: Tuple[Tensor, Tensor], batch_idx, train=True):
        x, y = batch
        y_hat = self.forward(x)
        predictions = y_hat.argmax(1)
        if not self.negative_labels:
            loss = F.cross_entropy(y_hat, y)
            accuracy = torch.equal(y, predictions).float().mean()
        else:
            negative_mask = torch.less(y, 0)
            if (~negative_mask).all():
                loss = F.cross_entropy(y_hat, y)
                accuracy = (y == predictions).float().mean()
            else:
                y[negative_mask] = -(y[negative_mask] + 1)
                losses = F.cross_entropy(y_hat, y, reduction='none')
                # True -> -1
                # False -> self.training_multiplier
                loss = torch.mean(
                    losses * torch.add(negative_mask * (-1 - self.training_multiplier), self.training_multiplier))

                # when dealing with negative labels the accuracy is the mean of the accuracy from the positive labels
                # and the error rate on the negative labels
                # noinspection PyTypeChecker
                accuracy = torch.cat([(y[~negative_mask] == predictions[~negative_mask]),
                                      (y[negative_mask] != predictions[negative_mask])]).float().mean()

        name = f'{"train" if train else "val"}'

        self.log(f'{name}/loss', loss.item())
        self.log(f'{name}/accuracy', accuracy.item())
        return loss

    def training_step(self, batch, batch_idx):
        return self.train_val_step(batch, batch_idx, train=True)

    def validation_step(self, batch, batch_idx):
        return self.train_val_step(batch, batch_idx, train=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
