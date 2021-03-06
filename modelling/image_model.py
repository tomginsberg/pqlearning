from typing import Any, Tuple, Dict, Set

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch import nn
from torchmetrics import MeanMetric as AverageMeter

from .lenet import LeNet
from .mlp import ImageMLP, ImageLinear
from .losses import ce_negative_labels_from_logits
from utils.generic import update_functional
from modelling.classifier import Classifier


class ImageModel(Classifier):
    def __init__(self, out_features: int = 10, in_channels: int = 3, arch='resnet18', pretrained=False, lr=0.01,
                 optim='Adam', gamma=.2, step_size=20, scheduler=None, check_negative_labels=True, channels_last=False,
                 **kwargs):
        """

        Args:
            out_features:   Number of output classes
            in_channels:   Number of input channels
            arch:   Architecture to use
            pretrained:   Whether to use pretrained weights
            **kwargs:
        """
        super().__init__(lr=lr, optim=optim, gamma=gamma, step_size=step_size, scheduler=scheduler,
                         check_negative_labels=check_negative_labels, **kwargs)
        self.save_hyperparameters()

        arch = arch.lower()
        if arch not in self.supported_architectures():
            raise ValueError(f'Architecture {arch} not supported')

        if 'resnet' in arch:
            # do some minor surgery on the resnet
            self.model = torchvision.models.resnet.__dict__[arch](pretrained=pretrained)
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
        elif arch == 'imagemlp':
            self.model = ImageMLP(in_channels=in_channels, out_features=out_features)
        elif arch == 'imagelinear':
            self.model = ImageLinear(in_channels=in_channels, out_features=out_features)

        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

    @staticmethod
    def supported_architectures() -> Set[str]:
        return {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'lenet', 'imagemlp', 'imagelinear'}

    def forward(self, x) -> Any:
        return self.model(x)


class CNNModule(pl.LightningModule):
    """
    Base class for CNN models
    Default architecture is Resnet 18
    """

    def __init__(self,
                 out_features: int = 10,
                 in_channels: int = 3,
                 negative_labels=False,
                 use_random_vectors=False,
                 training_multiplier=1,
                 learning_rate=1e-3,
                 weight_decay=5e-4,
                 logging_prefix=None,
                 arch='resnet18',
                 optim='sgd',
                 schedule='cosine',
                 schedule_args=None,
                 pretrained=False,
                 ckp=None,
                 n_train=None,
                 n_test=None,
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

        self.model = ImageModel(out_features=out_features, in_channels=in_channels, arch=arch,
                                pretrained=pretrained)

        if ckp is not None:
            if isinstance(ckp, str):
                self.load_state_dict(torch.load(ckp)['state_dict'])
            elif isinstance(ckp, nn.Module):
                self.load_state_dict(ckp.state_dict())
        # these are parameters useful for rejectron
        self.negative_labels = negative_labels
        self.use_random_vectors = use_random_vectors
        self.training_multiplier = 1 / training_multiplier

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # logging tools
        self.logging_prefix = logging_prefix
        self.train_accuracy = AverageMeter()
        self.val_accuracy = AverageMeter()

        self.p_accuracy = AverageMeter()
        self.q_accuracy = AverageMeter()
        self.n_p = n_train
        self.n_q = n_test
        self.optim, self.schedule, self.schedule_args = optim, schedule, schedule_args

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
                loss = F.cross_entropy(y_hat, y)
                accuracy = (y == predictions).float().mean()
            # if not, flip back the labels but negate the loss term for each negative label
            else:
                # y[negative_mask] = -(y[negative_mask] + 1)
                # losses = F.cross_entropy(y_hat, y, reduction='none')
                # # True -> -1
                # # False -> self.training_multiplier
                # loss = torch.mean(
                #     losses * torch.add(negative_mask * (-1 - self.training_multiplier), self.training_multiplier))

                loss = ce_negative_labels_from_logits(y_hat, y, alpha=self.training_multiplier,
                                                      use_random_vectors=self.use_random_vectors)
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
        # self.log(f'{name}/loss', loss.item())
        # self.log(f'{name}/accuracy', acc)
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
        if self.optim == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                                    momentum=0.9)
        else:
            raise ValueError(f'got {self.optim=}, use adam/sgd or update this code')
        if self.schedule is None:
            return optim

        update_sch_arg = update_functional(self.schedule_args)
        if self.schedule == 'cosine':
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=update_sch_arg(200, 'T_max'))
        elif self.schedule == 'step':
            schedule = torch.optim.lr_scheduler.StepLR(optim,
                                                       step_size=update_sch_arg(15, 'step_size'),
                                                       gamma=update_sch_arg(0.1, 'gamma')
                                                       )
        else:
            raise ValueError(f'got {self.schedule=}, use cosine/step or update this code')
        return [optim], [schedule]

    def training_epoch_end(self, outputs):
        self.log('train_epoch/accuracy', self.train_accuracy.compute())
        self.train_accuracy.reset()

        # negative labels implies that P and Q accuracies exist
        if self.negative_labels:
            # percentage of elements in P successfully classified to the label given
            p_acc = self.p_accuracy.compute()
            # percentage of elements in Q successfully classified differently from the label given
            q_acc = self.p_accuracy.compute()
            if self.n_p is not None and self.n_q is not None:
                s_metric = p_acc * self.n_p + (q_acc * self.n_q) / (self.n_q + 1)
                self.log('train_epoch/s_metric', s_metric)
            self.log('train_epoch/p_accuracy', p_acc)
            self.log('train_epoch/q_accuracy', q_acc)
            self.q_accuracy.reset(), self.p_accuracy.reset()

    def validation_epoch_end(self, outputs):
        self.log('val_epoch/accuracy', self.val_accuracy.compute())
        self.val_accuracy.reset()
