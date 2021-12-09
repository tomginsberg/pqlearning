import re
from glob import glob

import pandas as pd

from datasets import MnistDataModule
from experiments.rejectron.mnist import param_sets
from experiments.rejectron.rejectron import RejectronClassifier
from modelling import CNN
from modelling.pretrained import lenet_trained_on_mnist

# os.chdir(f'{os.environ["HOME"]}/pqlearning')
runs = glob('checkpoints/rejectron_mnist_sh-img_l_nT-*_seed-*')
nT_seed = [[int(re.match(re.compile('.*nT-(\d+)_seed-(\d).*'), x)[y]) for y in (1, 2)] for x in runs]
SHIFT = 'img_l'
model = lenet_trained_on_mnist().cuda(2)
if __name__ == '__main__':
    stats = []
    for run, (n_test, test_seed) in zip(runs, nT_seed):
        mnist = MnistDataModule(shift_transform_type='natural',
                                batch_size=512,
                                test_seed=test_seed,
                                train_val_seed=42,
                                test_sample_fraction=n_test / 10000,
                                fashion_mnist=False,
                                **param_sets[SHIFT])

        checkpoints = sorted(
            glob(f'{run}/c*/*ckpt'),
            # sort by model num in "checkpoints/rejectron_mnist/c_(model num)"
            key=lambda x: int(x.split('/')[2].split('_')[-1])
        )

        hS = RejectronClassifier(
            h=model,
        )
        hS.load_C_from_checkpoints(checkpoints, CNN)
        stats.append(hS.compute_accuracy_and_rejection_on_all(mnist) | {'n_test': n_test, 'test_seed': test_seed})
        print(stats[-1])

    pd.DataFrame(stats).to_csv('rejecton_mnist_img_l.csv', index=False)
