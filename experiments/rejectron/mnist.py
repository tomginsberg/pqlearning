from datasets import MnistDataModule
from experiments.rejectron.rejectron import RejectronClassifier, NoDataError, CreateRejectronModel
from modelling import CNNModule
from modelling.pretrained import lenet_trained_on_mnist
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

param_sets = {
    'img_l': dict(test_transform_rate=.5,
                  rotation=(45, 90),
                  crop=.5,
                  distortion=.8),
    'none': dict(test_transform_rate=0)
}
SHIFT = 'img_l'
initialization_strategy = 'base'
patience = 3


class RejectronMNISTCNN(CreateRejectronModel):
    def create(self, training_multiplier: float, logging_prefix: str, n_train: int, n_test: int):
        return CNNModule(
            in_channels=1, negative_labels=True,
            training_multiplier=training_multiplier,
            n_train=n_train,
            n_test=n_test,
            learning_rate=1e-3,
            weight_decay=0,
            logging_prefix=logging_prefix,
            arch='lenet'
        ).cuda(2)


if __name__ == '__main__':
    model = lenet_trained_on_mnist().cuda(device=2)
    stats_collection = []
    for n_test in (10000,):
        for test_seed in range(10):

            mnist = MnistDataModule(shift_transform_type='natural',
                                    batch_size=512,
                                    test_seed=test_seed,
                                    train_val_seed=42,
                                    test_sample_fraction=n_test / 10000,
                                    fashion_mnist=False,
                                    **param_sets[SHIFT])

            hS = RejectronClassifier(
                h=model,
                pq=mnist,
                create_model=RejectronMNISTCNN(),
                run_name=f'rejectron_mnist_sh-{SHIFT}_nT-{n_test}_seed-{test_seed}'
            )

            max_rej = 0
            count = 0
            stats = None
            for iteration in range(25):
                try:
                    stats = hS.train_next_c(
                        batch_size=512,
                        gpus=[2],
                        max_epochs=50,
                        initialization_strategy=None,
                        early_stopping=True
                    )
                    if x := stats['test_rej'] > max_rej:
                        max_rej = x
                        count = 0
                    else:
                        count += 1
                    if count > patience:
                        print(f'Rejection Rate has not increased in {count} round, stopping')
                        break
                except NoDataError:
                    print(f'Stopping run {n_test=} early on {iteration=}: No test data left!')
                    break

            if stats is not None:
                stats_collection.append({**stats, 'n_test': n_test, 'test_seed': test_seed})

    print(f'Job Completed\n{"-" * 60}')
    print(stats_collection)
    df = pd.DataFrame(stats_collection)
    df.to_csv(f'rejecton_mnist_{SHIFT}.csv', index=False)
