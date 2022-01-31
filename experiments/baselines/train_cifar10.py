import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datasets.cifar10 import CIFAR10DataModule
from modelling.image_model import ImageModel

if __name__ == '__main__':
    for seed in range(10):
        pl.seed_everything(seed, workers=True)

        RUN_NAME = f'cifar10_resnet18_{seed=}'
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=200,
            deterministic=True,
            callbacks=[EarlyStopping(monitor="val_acc", min_delta=0.00, patience=50, verbose=True, mode='max'),
                       ModelCheckpoint(dirpath=f'checkpoints/cifar/baselines/{RUN_NAME}',
                                       monitor='val_acc',
                                       save_top_k=1,
                                       verbose=True,
                                       mode='max',
                                       ),
                       LearningRateMonitor(),
                       ],
            logger=WandbLogger(project="pqlearning", offline=False, name=RUN_NAME),
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
        )
        model = ImageModel(in_channels=3, out_features=10, arch='resnet18', pretrained=False, lr=0.1, optim='SGD',
                           step_size=200, scheduler='cos', weight_decay=5e-4, momentum=0.9)
        # model = ImageModel.load_from_checkpoint(f'checkpoints/cifar/{RUN_NAME}/epoch=19-step=1959.ckpt')
        # model.lr = 0.008


        datamodule = CIFAR10DataModule(batch_size=128)

        trainer.fit(model, datamodule=datamodule)

