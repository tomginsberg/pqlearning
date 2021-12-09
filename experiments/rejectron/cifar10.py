from datasets.cifar10 import CIFAR10DataModule
from experiments.rejectron.rejectron import RejectronClassifier, NoDataError, CreateRejectronModel
from modelling import CNN
from modelling.pretrained import resnet18_trained_on_cifar10
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

initialization_strategy = 'base'
patience = 4
SHIFT = False


class RejectronCIFARCNN(CreateRejectronModel):
    def create(self, training_multiplier: float, logging_prefix: str, n_train: int, n_test: int):
        return CNN(
            in_channels=3, negative_labels=True,
            training_multiplier=training_multiplier,
            n_train=n_train,
            n_test=n_test,
            learning_rate=0.1,
            optim='sgd',
            weight_decay=5e-4,
            schedule='cosine',
            logging_prefix=logging_prefix,
            arch='resnet18'
        ).cuda(2)


if __name__ == '__main__':
    model = resnet18_trained_on_cifar10().cuda(device=2)
    stats_collection = []
    for n_test in (10, 50, 100, 500, 1000, 5000, 10000):
        for test_seed in range(10):

            mnist = CIFAR10DataModule(
                root='/voyager/datasets/',
                batch_size=512,
                test_seed=test_seed,
                test_samples=n_test,
                shift_types=('frost', 'fog', 'snow'),
                shift_severity_range=(3, 5),
                unshifted_test=not SHIFT
            )

            hS = RejectronClassifier(
                h=model,
                pq=mnist,
                create_model=RejectronCIFARCNN(),
                run_name=f'rejectron_cifar10/c{SHIFT}_nT-{n_test}_seed-{test_seed}'
                , dryrun=False
            )

            max_rej = 0
            count = 0
            stats = None
            for iteration in range(25):
                try:
                    stats = hS.train_next_c(
                        batch_size=512,
                        gpus=[2],
                        max_epochs=30,
                        initialization_strategy='base',
                        early_stopping=True
                    )

                    # early a stopping counter
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
    df.to_csv(f'rejecton_cifar10_test.csv', index=False)
