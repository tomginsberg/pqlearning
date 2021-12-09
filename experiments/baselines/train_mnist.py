import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datasets import MnistDataModule
from modelling import CNN

pl.seed_everything(42, workers=True)

RUN_NAME = 'mnist_baseline_resnet18'
trainer = pl.Trainer(
    auto_select_gpus=True,
    gpus=[2],
    max_epochs=50,
    deterministic=True,
    callbacks=[EarlyStopping(monitor="val_epoch/accuracy", min_delta=0.00, patience=5, verbose=True, mode='max'),
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
model = CNN(in_channels=1, out_features=10, arch='resnet18')
if __name__ == '__main__':
    trainer.fit(model, datamodule=MnistDataModule(fashion_mnist=False))
