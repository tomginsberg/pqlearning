import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datasets.coil import Coil100Module
from modelling import CNNModule

pl.seed_everything(42, workers=True)

RUN_NAME = 'coil_baseline_linear'
trainer = pl.Trainer(
    auto_select_gpus=True,
    gpus=[2],
    max_epochs=50,
    deterministic=True,
    callbacks=[EarlyStopping(monitor="val_epoch/accuracy", min_delta=0.00, patience=10, verbose=True, mode='max'),
               ModelCheckpoint(dirpath=f'checkpoints/{RUN_NAME}',
                               monitor='val_epoch/accuracy',
                               save_top_k=1,
                               verbose=True,
                               mode='max',
                               )],
    logger=WandbLogger(project="pqlearning", offline=False, name=RUN_NAME),
    num_sanity_val_steps=0,
    check_val_every_n_epoch=1,
    log_every_n_steps=1
)
model = CNNModule(in_channels=3, out_features=100, arch='image-linear')
if __name__ == '__main__':
    trainer.fit(model, datamodule=Coil100Module(split=1, batch_size=225))
