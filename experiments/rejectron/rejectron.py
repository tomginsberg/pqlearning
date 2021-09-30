from glob import glob
from typing import List, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class RejectronDataset(Dataset):
    def __init__(self, base_dataset: pl.LightningDataModule,
                 base_classifier: Callable[[torch.Tensor], torch.Tensor],
                 selective_classifier: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
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

        # we should have a weight of (n_test + 1) : 1 for training vs shifted-test examples seen while training
        self.train_multiplier = self.n_test * (self.n_test + 1) / self.n_train

    def label_training_data(self):
        with torch.no_grad():
            train_images, train_pseudo_labels = [], []
            for batch, _ in tqdm(self.base_dataset.train_dataloader()):
                batch = batch.cuda()
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
                batch = batch.cuda()
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
                 pq: pl.LightningDataModule,
                 create_model: Callable[[float], pl.LightningModule],
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
        self.create_model = create_model
        self.run_name_base = run_name

    def train_h(self):
        raise NotImplementedError('Please provide a pretrained h')

    def train_next_c(self, batch_size=128):
        trainer = pl.Trainer(
            auto_select_gpus=True,
            gpus=1,
            max_epochs=20,
            deterministic=True,
            callbacks=[EarlyStopping(monitor="train/accuracy", min_delta=0.00, patience=3, verbose=True, mode='max'),
                       ModelCheckpoint(dirpath=f'checkpoints/{self.run_name_base}/c_{len(self.C)}',
                                       monitor='train/accuracy',
                                       save_top_k=1,
                                       verbose=True,
                                       mode='max',
                                       )],
            logger=WandbLogger(project="pqlearning", offline=False, name=f'{self.run_name_base}/c_{len(self.C)}'),
            num_sanity_val_steps=0,
            check_val_every_n_epoch=9999,
        )

        dataset = RejectronDataset(base_dataset=self.pq, base_classifier=self.h,
                                   selective_classifier=self.selective_classify)
        c: pl.LightningModule = self.create_model(dataset.train_multiplier)
        trainer.fit(
            c, train_dataloader=DataLoader(
                dataset, shuffle=True, batch_size=batch_size
            )
        )
        ckp_path = glob(f'checkpoints/{self.run_name_base}/c_{len(self.C)}/*.ckp')
        if len(ckp_path) > 0:
            c = c.load_from_checkpoint(checkpoint_path=ckp_path)
            print(f'Successfully loaded {ckp_path}.')
        self.C.append(c.cuda())

    def selective_classify(self, x):
        y = torch.argmax(self.h(x), dim=1)
        for c in self.C:
            mask = torch.argmax(c(x), dim=1) != y
            y[mask] = -1
        return y


if __name__ == '__main__':
    from models.pretrained import resnet18_trained_on_mnist
    from models import CNN
    from datasets import MnistDataModule

    model = resnet18_trained_on_mnist().cuda()
    mnist = MnistDataModule(shift_transform_type='natural', test_transform_rate=.5, rotation=20, crop=.3, distortion=.1,
                            batch_size=256)
    hS = RejectronClassifier(
        h=model,
        pq=mnist,
        create_model=lambda training_multiplier: CNN(in_channels=1, negative_labels=True,
                                                     training_multiplier=training_multiplier,
                                                     learning_rate=0.1 / training_multiplier).cuda()
    )
    hS.train_next_c(batch_size=256)
    hS.train_next_c(batch_size=256)
