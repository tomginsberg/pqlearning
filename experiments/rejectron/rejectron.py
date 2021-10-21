from glob import glob
from typing import List, Callable, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb


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
        # we should have a weight of (n_test + 1) : 1 for training vs shifted-test examples seen while training
        self.train_multiplier = self.n_test * (self.n_test + 1) / self.n_train

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


class RejectronClassifier:
    def __init__(self,
                 h: nn.Module,
                 pq: Optional[pl.LightningDataModule] = None,
                 create_model: Optional[Callable[[float, str], pl.LightningModule]] = None,
                 run_name='rejectron_mnist',
                 ):
        # base classifier h: X -> Y
        self.h = h

        # datasets:
        #   {(x1, y1), ..., (xn, yn)} ~ p
        #   {x1, ..., xn} ~ q
        # pq should be a LightningDataModule with p=pq.test_dataloader() and q=pq.train_dataloader()
        self.pq = pq
        self.C: List[nn.Module] = []
        self.C_checkpoints: List[str] = []
        self.create_model = create_model
        self.run_name_base = run_name
        self.seed = 0

    def train_h(self):
        raise NotImplementedError('Please provide a pretrained h')

    def train_next_c(self, batch_size=128, gpus=[2], max_epochs=30, early_stopping=False,
                     initialization_strategy=None):

        dataset = RejectronDataset(base_dataset=self.pq, base_classifier=self.h,
                                   selective_classifier=self.selective_classify)

        pl.seed_everything(seed=self.seed)
        self.seed += 1

        c: pl.LightningModule = self.create_model(dataset.train_multiplier, f'c_{len(self.C)}')

        if initialization_strategy is not None:
            assert initialization_strategy in {'base', 'previous'}
            if initialization_strategy == 'base':
                c.load_state_dict(self.h.state_dict())
            else:
                if len(self.C) == 0:
                    print('Initializing c_1 from h')
                    c.load_state_dict(self.h.state_dict())
                else:
                    print(f'Initializing c_{len(self.C) + 1} from c_{len(self.C)}')
                    c.load_state_dict(self.C[-1].state_dict())

        callbacks = [
            ModelCheckpoint(dirpath=f'checkpoints/{self.run_name_base}/c_{len(self.C)}',
                            monitor='train_epoch/accuracy',
                            save_top_k=1,
                            verbose=True,
                            mode='max',
                            save_on_train_epoch_end=True
                            )]
        if early_stopping:
            callbacks = [
                EarlyStopping(monitor='train_epoch/accuracy', min_delta=0.00, patience=5, verbose=True, mode='max'),
                callbacks[0]
            ]

        trainer = pl.Trainer(
            auto_select_gpus=True,
            gpus=gpus,
            max_epochs=max_epochs,
            deterministic=True,
            callbacks=callbacks,
            logger=WandbLogger(project="pqlearning", offline=False, name=f'{self.run_name_base}'),
            num_sanity_val_steps=0,
            # this is arbitrarily high as we do not perform validation when training rejectron
            check_val_every_n_epoch=99999,
        )

        trainer.fit(
            c, train_dataloaders=DataLoader(
                dataset, shuffle=True, batch_size=batch_size
            )
        )

        wandb.log({'n_test': dataset.n_test})

        ckp_path = glob(f'checkpoints/{self.run_name_base}/c_{len(self.C)}/*.ckpt')
        if len(ckp_path) == 1:
            c = c.load_from_checkpoint(checkpoint_path=ckp_path[0])
            print(f'Successfully loaded {ckp_path[0]}.')
            self.C_checkpoints.append(ckp_path[0])

        elif len(ckp_path) != 1:
            raise RuntimeError(f'Multiple/No Checkpoints found: {ckp_path}')
        self.C.append(c.cuda(device=gpus[0]))

    def selective_classify(self, x):
        y = torch.argmax(self.h(x), dim=1)
        for c in self.C:
            mask = torch.argmax(c(x), dim=1) != y
            y[mask] = -1
        return y


if __name__ == '__main__':
    from modelling.pretrained import lenet_trained_on_coil_p1
    from modelling import CNN
    from datasets.coil import Coil100Module

    model = lenet_trained_on_coil_p1().cuda(device=2)
    coil = Coil100Module(split=1)
    hS = RejectronClassifier(
        h=model,
        pq=coil,
        create_model=lambda training_multiplier, logging_prefix:
        CNN(in_channels=3,
            out_features=100,
            negative_labels=True,
            training_multiplier=training_multiplier,
            learning_rate=1e-2 / training_multiplier,
            logging_prefix=logging_prefix,
            arch='lenet',
            # ckp='checkpoints/coil_baseline_lenet/epoch=35-step=287.ckpt'
            ).cuda(),
        run_name='rejectron_coil'
    )

    hS.train_next_c(batch_size=64, gpus=[2], max_epochs=75, )
