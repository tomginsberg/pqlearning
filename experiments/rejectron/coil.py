from modelling.pretrained import lenet_trained_on_coil_p1, linear_trained_on_coil_p1
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
        learning_rate=1 / training_multiplier,
        logging_prefix=logging_prefix,
        arch='lenet',
        ).cuda(),
    run_name='rejectron_coil_p1_lenet'
)

if __name__ == '__main__':
    for _ in range(64):
        hS.train_next_c(batch_size=225, gpus=[2], max_epochs=50, early_stopping=False, initialization_strategy='base')
