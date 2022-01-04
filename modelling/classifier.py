import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from abc import abstractmethod


class Classifier(pl.LightningModule):
    def __init__(self, lr=1e-3, optim='Adam', scheduler='step', gamma=.2, step_size=20, weight_decay=1e-4, momentum=0.9,
                 check_negative_labels=True,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.lr = lr
        self.optim_name = optim
        self.gamma = gamma
        self.step_size = step_size
        self.weight_decay = weight_decay
        self.schedule = scheduler
        self.momentum = momentum
        self.check_negative_labels = check_negative_labels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, *args, **kwargs) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        self.train_acc(y_hat, y)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, *args, **kwargs) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        if self.check_negative_labels:
            y[y < 0] = -y[y < 0] - 1
        self.val_acc(y_hat, y)
        loss = self.loss(y_hat, y)
        return loss

    def validation_epoch_end(self, outputs: torch.Tensor):
        avg_loss = torch.tensor(outputs).mean()
        avg_acc = self.val_acc.compute()
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)
        self.val_acc.reset()

    def training_epoch_end(self, outputs: torch.Tensor):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = self.train_acc.compute()
        self.log('train_loss', avg_loss)
        self.log('train_acc', avg_acc)
        self.train_acc.reset()

    def configure_optimizers(self):
        params = dict(lr=self.lr, weight_decay=self.weight_decay)
        if self.optim_name == 'SGD':
            params['momentum'] = self.momentum

        optim = torch.optim.__dict__[self.optim_name](self.model.parameters(), **params)

        # make it easy to turn off scheduling
        if self.schedule == 'step':
            if isinstance(self.step_size, int):
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=self.gamma, step_size=self.step_size)
            elif isinstance(self.step_size, list):
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, gamma=self.gamma,
                                                                 milestones=self.step_size)
            else:
                raise ValueError('step_size must be either int or list, got {}'.format(type(self.step_size)))

        elif self.schedule == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=self.gamma)
        elif 'cos' in self.schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=self.step_size)
        elif self.schedule == 'none' or self.schedule is None:
            return optim
        else:
            raise ValueError('{self.schedule=} is not recognized')

        return [optim], [scheduler]
