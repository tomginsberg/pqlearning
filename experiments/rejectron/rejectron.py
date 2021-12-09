from abc import ABC
from glob import glob
from typing import List, Callable, Optional, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from torchmetrics import Accuracy, AverageMeter


class NoDataError(Exception):
    pass


class RejectronDataset(Dataset):
    def __init__(self, base_dataset: pl.LightningDataModule,
                 base_classifier: Callable[[torch.Tensor], torch.Tensor],
                 selective_classifier: Callable[[torch.Tensor], torch.Tensor],
                 cuda_device=2):
        super().__init__()
        self.device = cuda_device
        self.base_dataset = base_dataset
        self.selective_classifier = selective_classifier
        self.base_classifier = base_classifier
        print('Inferring pseudo labels on training set')
        train_imgs, train_labels = self.label_training_data()
        print('Inferring pseudo labels on test set')
        test_imgs, test_labels = self.label_testing_data()

        self.all_imgs = torch.cat((train_imgs, test_imgs))
        self.all_labels = torch.cat((train_labels, test_labels))
        self.n_train, self.n_test = len(train_labels), len(test_labels)
        print(f'Size of training set: {self.n_train}')
        print(f'Size of testing set: {self.n_test}')
        if self.n_test == 0:
            raise NoDataError()
        # we should have a weight of 1 : 1/(n_train + 1) for training vs shifted-test examples seen while training
        self.train_multiplier = (self.n_train + 1)

    def update_selective_classifier(self, selective_classifier):
        self.selective_classifier = selective_classifier
        print('Inferring pseudo labels on test set')
        test_imgs, test_labels = self.label_testing_data()
        self.all_imgs = torch.cat((self.all_imgs[:self.n_train], test_imgs))
        self.all_labels = torch.cat((self.all_labels[:self.n_train], test_labels))
        self.n_test = len(test_labels)
        print(f'Size of testing set: {self.n_test}')
        if self.n_test == 0:
            raise NoDataError()
        self.train_multiplier = (self.n_train + 1)

    def label_training_data(self):
        with torch.no_grad():
            train_images, train_pseudo_labels = [], []
            for batch, _ in tqdm(self.base_dataset.train_dataloader()):
                batch = batch.cuda(device=self.device)
                y_hat = self.base_classifier(batch).cpu().argmax(1)
                train_images.append(batch)
                train_pseudo_labels.append(y_hat)

            train_images = torch.cat(train_images)
            train_pseudo_labels = torch.cat(train_pseudo_labels)
            return train_images, train_pseudo_labels

    def label_testing_data(self):
        with torch.no_grad():
            test_images, test_pseudo_labels = [], []
            for batch, _ in tqdm(self.base_dataset.test_dataloader()):
                batch = batch.cuda(device=self.device)
                y_hat = self.selective_classifier(batch).cpu()
                test_images.append(batch[y_hat != -1])
                test_pseudo_labels.append(-y_hat[y_hat != -1] - 1)

            test_pseudo_labels = torch.cat(test_pseudo_labels)
            test_images = torch.cat(test_images)
            return test_images, test_pseudo_labels

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        return self.all_imgs[idx], self.all_labels[idx]


class CreateRejectronModel(ABC):
    def create(self, training_multiplier: float, logging_prefix: str, n_train: int, n_test: int):
        raise NotImplementedError


class RejectronClassifier:
    def __init__(self,
                 h: nn.Module,
                 pq: Optional[pl.LightningDataModule] = None,
                 create_model: Optional[CreateRejectronModel] = None,
                 run_name='unnamed',
                 device=2,
                 dryrun=False,
                 ):
        # base classifier h: X -> Y
        self.h = h.eval()

        # datasets:
        #   {(x1, y1), ..., (xn, yn)} ~ p
        #   {x1, ..., xn} ~ q
        # pq should be a LightningDataModule with p=pq.test_dataloader() and q=pq.train_dataloader()
        self.pq = pq
        self.C: List[nn.Module] = []
        self.C_checkpoints: List[str] = []
        if create_model is not None:
            self.create_model = create_model.create
        else:
            self.create_model = None
        self.run_name_base = run_name
        self.seed = 0
        self.rejectron_dataset = None
        self.device = device
        self.dryrun = dryrun

    def load_C_from_checkpoints(self, checkpoints: List[str], model_cls: type(pl.LightningModule)):
        self.C_checkpoints = checkpoints
        self.C = [model_cls.load_from_checkpoint(ck).cuda(self.device).eval() for ck in checkpoints]

    def train_h(self):
        raise NotImplementedError('Please provide a pretrained h')

    def compute_accuracy_and_rejection_on_all(self, datamodule: pl.LightningDataModule) -> Dict[str, float]:
        tr = {'train_' + k: v for k, v in self.compute_accuracy_and_rejection(datamodule.train_dataloader()).items()}
        va = {'val_' + k: v for k, v in self.compute_accuracy_and_rejection(datamodule.val_dataloader()).items()}
        te = {'test_' + k: v for k, v in self.compute_accuracy_and_rejection(datamodule.test_dataloader()).items()}
        return tr | va | te

    def compute_accuracy_and_rejection(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        with torch.no_grad():
            reject, accuracy = AverageMeter(), Accuracy()
            for x, y in tqdm(dataloader):
                j = self.selective_classify(x.cuda(self.device)).cpu()
                m = j != -1
                reject.update(~m)
                if not (~m).all():
                    accuracy.update(j[m], y[m])
            if not accuracy.mode:
                acc = 1
            else:
                acc = accuracy.compute().item()

        return {'acc': acc, 'rej': reject.compute().item()}

    def train_next_c(self, batch_size=128, gpus=[2], max_epochs=30, early_stopping=False,
                     initialization_strategy=None) -> Dict[str, float]:
        assert isinstance(self.pq,
                          pl.LightningDataModule), \
            f'to train next c make self.pq a lightning data module not {type(self.pq)=}'

        if self.rejectron_dataset is None:
            self.rejectron_dataset = RejectronDataset(base_dataset=self.pq, base_classifier=self.h,
                                                      selective_classifier=self.selective_classify)
        else:
            self.rejectron_dataset.update_selective_classifier(self.selective_classify)

        pl.seed_everything(seed=self.seed)
        self.seed += 1

        c: pl.LightningModule = self.create_model(training_multiplier=self.rejectron_dataset.train_multiplier,
                                                  logging_prefix=f'c_{len(self.C)}',
                                                  n_train=self.rejectron_dataset.n_train,
                                                  n_test=self.rejectron_dataset.n_test)

        if initialization_strategy is not None:
            assert initialization_strategy in {'base', 'previous'}
            if initialization_strategy == 'base':
                c.load_state_dict(self.h.state_dict().copy())
            else:
                if len(self.C) == 0:
                    print('Initializing c_1 from h')
                    c.load_state_dict(self.h.state_dict().copy())
                else:
                    print(f'Initializing c_{len(self.C) + 1} from c_{len(self.C)}')
                    c.load_state_dict(self.C[-1].state_dict().copy())

        c.train()
        callbacks = []

        if early_stopping:
            callbacks.append(
                EarlyStopping(monitor='train_epoch/s_metric', min_delta=0.00, patience=15, verbose=True, mode='max')
            )

        if not self.dryrun:
            callbacks.append(ModelCheckpoint(dirpath=f'checkpoints/{self.run_name_base}/c_{len(self.C)}',
                                             monitor='train_epoch/s_metric',
                                             save_top_k=1,
                                             verbose=True,
                                             mode='max',
                                             save_on_train_epoch_end=True,
                                             ))

        trainer = pl.Trainer(
            auto_select_gpus=True,
            gpus=gpus,
            max_epochs=max_epochs,
            deterministic=True,
            callbacks=callbacks,
            logger=WandbLogger(project="pqlearning", offline=self.dryrun, name=f'{self.run_name_base}'),
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
        )

        trainer.fit(
            c, train_dataloaders=DataLoader(
                self.rejectron_dataset, shuffle=True, batch_size=batch_size, num_workers=0
            )
        )
        c.eval()
        if not self.dryrun:
            ckp_path = glob(f'checkpoints/{self.run_name_base}/c_{len(self.C)}/*.ckpt')
            if len(ckp_path) == 1:
                c = c.load_from_checkpoint(checkpoint_path=ckp_path[0])
                c.eval()
                print(f'Successfully loaded {ckp_path[0]}.')
                self.C_checkpoints.append(ckp_path[0])

            elif len(ckp_path) != 1:
                raise RuntimeError(f'Multiple/No Checkpoints found: {ckp_path}')

        self.C.append(c.cuda(device=gpus[0]))

        print(f'Computing Acc/Rej stats for h|S_{len(self.C) + 1}')

        # a hack to turn off data augmentation for certain datasets
        # this should be fixed by making a UpdatableDatamodule type
        if hasattr(self.pq, 'update_test_transform'):
            self.pq.update_test_transform(train=False)

        acc_rej_stats = [self.compute_accuracy_and_rejection(x) for x in
                         [self.pq.train_dataloader(), self.pq.val_dataloader(), self.pq.test_dataloader()]]

        if hasattr(self.pq, 'update_test_transform'):
            self.pq.update_test_transform(train=True)

        stats = {f'{x}_{cat}': acc_rej_stats[i][cat]
                 for i, x in enumerate(['train', 'val', 'test']) for cat in ['acc', 'rej']}
        for k, v in stats.items():
            print(f'{k}: {v}')
        wandb.log({**stats, 'iteration': self.seed + 1})

        return stats

    def selective_classify(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.argmax(self.h(x), dim=1)
        for c in self.C:
            mask = torch.argmax(c(x), dim=1) != y
            y[mask] = -1
        return y
