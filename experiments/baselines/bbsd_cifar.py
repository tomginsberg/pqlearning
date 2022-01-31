import pandas as pd
from pytorch_lightning import seed_everything
from scipy.stats import ks_2samp

from datasets.cifar10 import CIFAR10DataModule
from experiments.baselines.bbsd import generate_logits
from modelling import pretrained
from tqdm import tqdm
from scipy.stats import boschloo_exact
from glob import glob

def univariate_ks_test(p_logits, q_logits):
    pvals = [ks_2samp(p_logits[:, i], q_logits[:, i]).pvalue for i in tqdm(range(p_logits.shape[1]))]
    return pvals, min(pvals) < 0.05 / p_logits.shape[1]


if __name__ == '__main__':
    # define h

    h_collection = pretrained.resnet18_collection_trained_on_cifar10()
    dl = CIFAR10DataModule().val_dataloader()
    p_logits_ = [generate_logits(h, dl) for h in h_collection]

    seed_range = range(10)
    sample_range = [10, 100, 1000, 10000]
    df = pd.DataFrame()

    # define dataset
    for shift in (True, False):
        for model_seed, (h, p_logits) in enumerate(zip(h_collection, p_logits_)):
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
                        negative_labels=False
                    )
                    q_logits, q_acc = generate_logits(h, dm.test_dataloader(), compute_acc=True)
                    pvals, significant = univariate_ks_test(p_logits, q_logits)
                    result = dict(
                        pvals=pvals,
                        test_acc=q_acc.item(),
                        significant=significant,
                        test_samples=test_samples,
                        model_seed=model_seed,
                        test_seed=test_seed,
                        shift=shift
                    )
                    df = df.append(result, ignore_index=True)
                    print(df.iloc[-1])
                    df.to_csv('tables/bbsd_cifar_ensemble.csv', index=False)
                        # don't bother running this again if test_samples is 10000

    df.to_csv('tables/bbsd_cifar_ensemble.csv', index=False)
