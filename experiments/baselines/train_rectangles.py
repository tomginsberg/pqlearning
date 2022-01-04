import pytorch_lightning as pl

from datasets.rectangles import RectanglesModule
from modelling import MLP

if __name__ == '__main__':
    mlp = MLP(in_features=2, out_features=2, lr=0.02, hidden_layers=(16, 32, 64, 128, 32))
    trainer = pl.Trainer(
        auto_select_gpus=True,
        gpus=1,
        max_epochs=80,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                'checkpoints/rectangles',
                save_top_k=1,
                monitor='val_acc',
                mode='max',
                verbose=True,
                filename='rectangles_mlp_large-{epoch}-{val_acc:.2f}'
            ),
            pl.callbacks.EarlyStopping(monitor='val_acc', patience=5, mode='max', verbose=True)
        ],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    data = RectanglesModule()
    trainer.fit(mlp, datamodule=data)
    trainer.validate(mlp, data.train_dataloader(), verbose=True)
