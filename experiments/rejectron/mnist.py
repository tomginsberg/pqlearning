from modelling.pretrained import lenet_trained_on_mnist
from modelling import CNN
from datasets import MnistDataModule
from experiments.rejectron.rejectron import RejectronClassifier

model = lenet_trained_on_mnist().cuda(device=2)
mnist = MnistDataModule(shift_transform_type='natural', test_transform_rate=.5, rotation=(45, 90), crop=.2,
                        distortion=.8,
                        batch_size=256)

hS = RejectronClassifier(
    h=model,
    pq=mnist,
    create_model=lambda training_multiplier, logging_prefix: CNN(in_channels=1, negative_labels=True,
                                                                 training_multiplier=training_multiplier,
                                                                 learning_rate=1 / training_multiplier,
                                                                 logging_prefix=logging_prefix,
                                                                 arch='lenet').cuda(),
    run_name='rejectron_mnist_p5_r45-90_c2-d8_lenet'
)
for _ in range(64):
    hS.train_next_c(batch_size=512, gpus=[2], max_epochs=100, initialization_strategy='previous', early_stopping=True)
