from modelling.pretrained import lenet_trained_on_coil_p1
from modelling import CNN
from datasets.coil import Coil100Module
from experiments.rejectron.rejectron import RejectronClassifier

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
        learning_rate=4 / training_multiplier,
        logging_prefix=logging_prefix,
        arch='lenet',
        ckp='checkpoints/coil_baseline_lenet/epoch=35-step=287.ckpt'
        ).cuda(),
    run_name='rejectron_coil'
)

if __name__ == '__main__':
    for _ in range(64):
        hS.train_next_c(batch_size=150, gpus=[2], max_epochs=400, early_stopping=False)
