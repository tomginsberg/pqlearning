import pandas as pd

from datasets.pqmodule import PQModule
from datasets.cifar10 import CIFAR10DataModule
from detectron.utils import rejectron_trainer, train_rejectors
from modelling import pretrained
from pytorch_lightning import seed_everything

if __name__ == '__main__':
    # define h
    h = pretrained.resnet18_trained_on_cifar10()
    seed_range = range(10)
    sample_range = [10, 100, 1000, 10000]
    df = pd.DataFrame()
    # define dataset
    for shift in (False,):
        for test_samples in sample_range:
            for test_seed in seed_range:
                seed_everything(test_seed)
                # define a PQDataModule
                dm = CIFAR10DataModule(
                    test_seed=test_seed,
                    test_samples=test_samples,
                    shift_types=('frost', 'fog', 'snow'),
                    shift_severity_range=(3, 5),
                    unshifted_test=not shift,
                )
                pq = PQModule(
                    p=dm.train_dataset, p_prime=dm.val_dataset, q=dm.test_dataset,
                    batch_size=256, num_workers=96//2, drop_last=False
                )

                # get a default rejectron trainer
                trainer = rejectron_trainer(save_directory=f'cifar/rejectron_{test_seed=}_{test_samples=}_{shift=}',
                                            max_epochs=25,
                                            run_name=f'cifar_rej_{test_seed=}_{test_samples=}_{shift=}',
                                            dryrun=True, gpus=[1])

                # train !
                result = train_rejectors(pq=pq, h=h, trainer=trainer, num_rejectors=32,
                                         logfile=f'checkpoints/cifar/rejectron_{test_seed=}_{test_samples=}_{shift=}')

                result = result.iloc[-1].to_dict() | dict(test_samples=test_samples, test_seed=test_seed, shift=shift)
                df = df.append(result, ignore_index=True)
                df.to_csv('checkpoints/cifar/rejectron_results.csv', index=False)
                print(df)
