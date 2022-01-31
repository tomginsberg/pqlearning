from typing import List

import pytorch_lightning as pl
import torch
import torchvision
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from modelling.image_model import ImageModel

if __name__ == '__main__':

    MEAN = [122.5502, 102.1070, 91.2910]
    STD = [71.9679, 64.6399, 62.5419]

    BATCH_SIZE = 512

    loaders = {}
    for name in ['train', 'test', 'val']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, MEAN))),  # Note Cutout is done before normalization.
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
            torchvision.transforms.Normalize(MEAN, STD),
        ])

        # Create loaders
        loaders[name] = Loader(f'/voyager/datasets/CelebA/{name}.beton',
                               batch_size=BATCH_SIZE,
                               num_workers=96 // 2,
                               order=OrderOption.RANDOM,
                               drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline,
                                          'label': label_pipeline})

    RUN_NAME = 'attractive_young_baseline'
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=50,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_acc", min_delta=0.00, patience=10, verbose=True, mode='max'),
                   ModelCheckpoint(dirpath=f'checkpoints/celeb/baselines/{RUN_NAME}',
                                   monitor='val_acc',
                                   save_top_k=1,
                                   verbose=True,
                                   mode='max',
                                   ),
                   # LearningRateMonitor(),
                   ],
        logger= WandbLogger(project="pqlearning", offline=False, name=RUN_NAME),
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        # log_every_n_steps=1,
    )
    model = ImageModel(in_channels=3, out_features=2, arch='resnet18', pretrained=False, lr=0.1, optim='SGD',
                       step_size=50, scheduler='cos', weight_decay=5e-4, momentum=0.9, check_negative_labels=False,
                       channels_last=True)
    # model = model.to(memory_format=torch.channels_last)
    trainer.fit(model, train_dataloaders=loaders['train'], val_dataloaders=loaders['val'])
    from getpass import getpass