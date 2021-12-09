import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datasets.cifar10 import CIFAR10DataModule
from modelling import CNN

if __name__ == '__main__':
    pl.seed_everything(42, workers=True)

    RUN_NAME = 'cifar10_baseline_resnet50'
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=200,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_epoch/accuracy", min_delta=0.00, patience=50, verbose=True, mode='max'),
                   ModelCheckpoint(dirpath=f'checkpoints/{RUN_NAME}',
                                   monitor='val_epoch/accuracy',
                                   save_top_k=1,
                                   verbose=True,
                                   mode='max',
                                   )],
        logger=WandbLogger(project="pqlearning", offline=False, name=RUN_NAME),
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )
    model = CNN(in_channels=3, out_features=10, arch='resnet50', pretrained=True, optim='sgd', schedule='cosine',
                learning_rate=0.1)
    datamodule = CIFAR10DataModule(batch_size=512)

    trainer.fit(model, datamodule=datamodule)
