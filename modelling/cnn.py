from typing import Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch import nn
from torchmetrics import AverageMeter

from .lenet import LeNet
from .losses import ce_negative_labels_from_logits


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
                 learning_rate=1e-3,
                 logging_prefix=None,
                 arch='resnet18',
                 ckp=None,
                 **kwargs: Any):
        """

        :param out_features: Number of output features (default 10 for mnist)
        :param in_channels: Number of inout channels
        :param negative_labels: If true the loss will be maximized for entries with negative class labels
        this is useful for rejectron training
        :param training_multiplier:
        :param learning_rate:
        :param logging_prefix: log all entries with the form {logging_prefix}/train/loss
        :param arch: string name of arch (default resnet18) also supports lenet
        :param kwargs: arbitrary kwargs for compatibility
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # do some minor surgery on the resnet
        arch = arch.lower()  # ignore case
        if arch == 'resnet18':
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
        elif arch == 'lenet':
            self.model = LeNet(in_channels=in_channels, out_features=out_features)
        else:
            raise ValueError(f'arch ({arch}) is invalid')

        if ckp is not None:
            if isinstance(ckp, str):
                self.load_state_dict(torch.load(ckp)['state_dict'])
            elif isinstance(ckp, nn.Module):
                self.load_state_dict(ckp.state_dict())
        # these are parameters useful for rejectron
        self.negative_labels = negative_labels
        self.training_multiplier = training_multiplier

        self.learning_rate = learning_rate

        # logging tools
        self.logging_prefix = logging_prefix
        self.train_accuracy = AverageMeter()
        self.val_accuracy = AverageMeter()

        self.p_accuracy = AverageMeter()
        self.q_accuracy = AverageMeter()

    def forward(self, x) -> Tensor:
        return self.model(x)

    def train_val_step(self, batch: Tuple[Tensor, Tensor], batch_idx, train=True):
        x, y = batch
        y_hat = self.forward(x)
        predictions = y_hat.argmax(1)

        # set a placeholder value for the accuracy on P
        p_acc = -1
        if not self.negative_labels:
            loss = F.cross_entropy(y_hat, y)
            accuracy = (y == predictions).float().mean()
        else:
            negative_mask = torch.less(y, 0)
            # if no labels in the batch are negative proceed as normal
            if (~negative_mask).all():
                loss = self.training_multiplier * F.cross_entropy(y_hat, y)
                accuracy = (y == predictions).float().mean()
            # if not, flip back the labels but negate the loss term for each negative label
            else:
                # y[negative_mask] = -(y[negative_mask] + 1)
                # losses = F.cross_entropy(y_hat, y, reduction='none')
                # # True -> -1
                # # False -> self.training_multiplier
                # loss = torch.mean(
                #     losses * torch.add(negative_mask * (-1 - self.training_multiplier), self.training_multiplier))

                loss = ce_negative_labels_from_logits(y_hat, y, pos_weight=self.training_multiplier)
                # when dealing with negative labels the accuracy is the mean of the accuracy from the positive labels
                # and the error rate on the negative labels

                p_acc = (y[~negative_mask] == predictions[~negative_mask])

                y[negative_mask] = -(y[negative_mask] + 1)
                q_acc = (y[negative_mask] != predictions[negative_mask])

                # noinspection PyTypeChecker
                accuracy = torch.cat([
                    p_acc,
                    q_acc
                ]).float().mean()

                p_acc, q_acc = p_acc.float().mean(), q_acc.float().mean()
                self.p_accuracy.update(p_acc)
                self.q_accuracy.update(q_acc)

        # logging
        name = f'{"train" if train else "val"}'
        if self.logging_prefix is not None:
            name = self.logging_prefix + '/' + name
        acc = accuracy.item()
        self.log(f'{name}/loss', loss.item())
        self.log(f'{name}/accuracy', acc)
        if train:
            self.train_accuracy.update(acc)
            if p_acc == -1:
                # is p_acc has not been updated from the placeholder p_acc
                # is just the accuracy i.e no negative labels in this batch
                self.p_accuracy.update(acc)
        else:
            self.val_accuracy.update(acc)

        return loss

    def training_step(self, batch, batch_idx):
        return self.train_val_step(batch, batch_idx, train=True)

    def validation_step(self, batch, batch_idx):
        return self.train_val_step(batch, batch_idx, train=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_epoch_end(self, outputs):
        self.log('train_epoch/accuracy', self.train_accuracy.compute())
        self.train_accuracy.reset()
        if self.negative_labels:
            # negative labels implies that P and Q accuracies exist
            self.log('train_epoch/p_accuracy', self.p_accuracy.compute())
            self.log('train_epoch/q_accuracy', self.q_accuracy.compute())
            self.q_accuracy.reset(), self.p_accuracy.reset()

    def validation_epoch_end(self, outputs):
        self.log('val_epoch/accuracy', self.val_accuracy.compute())
        self.val_accuracy.reset()
