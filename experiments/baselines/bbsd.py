from torch.utils.data import DataLoader
import torch
import warnings
from tqdm import tqdm
from scipy.stats import ks_2samp
from modelling.pretrained import lenet_trained_on_mnist
from datasets import MnistDataModule
import pandas as pd
import numpy as np
import json


def generate_logits(model: torch.nn.Module, dataloader: DataLoader):
    device = model.device
    with torch.no_grad():
        return torch.cat([model(x.to(device)) for x, _ in dataloader], dim=0).cpu()


param_sets = {
    'img_l': dict(test_transform_rate=.5,
                  rotation=(45, 90),
                  crop=.5,
                  distortion=.8),
    'none': dict(test_transform_rate=0)
}
SHIFT = 'img_l'

warnings.filterwarnings("ignore")


def write_logits_to_json():
    p_logits = []
    q_logits = []
    model = lenet_trained_on_mnist().cuda(2)

    for n_test in (10, 50, 100, 500, 1000, 5000, 10000):
        print(f'{n_test=}')
        for test_seed in tqdm(range(10)):
            mnist = MnistDataModule(shift_transform_type='natural',
                                    batch_size=512,
                                    test_seed=test_seed,
                                    train_val_seed=42,
                                    test_sample_fraction=n_test / 10000,
                                    fashion_mnist=False,
                                    **param_sets['none'])

            p_logits.append({'logits': generate_logits(model, mnist.test_dataloader()).T.tolist(), 'n_test': n_test,
                             'test_seed': test_seed})

            mnist = MnistDataModule(shift_transform_type='natural',
                                    batch_size=512,
                                    test_seed=test_seed,
                                    train_val_seed=42,
                                    test_sample_fraction=n_test / 10000,
                                    fashion_mnist=False,
                                    **param_sets['img_l'])

            q_logits.append({'logits': generate_logits(model, mnist.test_dataloader()).T.tolist(), 'n_test': n_test,
                             'test_seed': test_seed})

    pd.DataFrame(p_logits).to_json('logits_mnist_none.json', orient='records')
    pd.DataFrame(q_logits).to_json('logits_mnist_img-l.json', orient='records')


def run_all_ks_tests(p='logits_mnist_none.json', q='logits_mnist_img-l.json'):
    p = pd.read_json(p)
    q = pd.read_json(q)
    n_test_p = p.n_test.unique()
    n_test_q = q.n_test.unique()
    res = {str((n_p, n_q)): [] for n_p in n_test_p for n_q in n_test_q}
    for rp in tqdm(p.iloc):
        for rq in q.iloc:
            res[str((rp.n_test, rq.n_test))].append(
                [ks_2samp(x, y).pvalue for x, y in zip(np.array(rp.logits), np.array(rq.logits))])

    res['(np, nq)'] = ''
    with open('bbsd_mnist.json', 'w') as f:
        json.dump(res, f)


def ks_test_batch(p, q):
    """
    p is the set of validation runs (10 runs x 10 classes x n samples)
    q is another set of runs (10 runs x 10 classes x k samples)
    """
    n = p.shape[-1]
    k = q.shape[-1]
    classes = p.shape[1]
    res = [[] for _ in range(classes)]

    for c in tqdm(range(classes)):
        # get the logits from p and q with class c
        pp = p[:, c]
        qq = q[:, c]
        for p_run in pp:
            for q_run in qq:
                res[c].append(ks_2samp(p_run, q_run).pvalue)
    return (n, k), np.array(res).T.tolist()


def no_shift_test(p='logits_mnist_none.json', val_sizes=(10, 10000)):
    p = pd.read_json(p)

    val_sets = [np.array(p[p.n_test == x].logits.tolist()) for x in val_sizes]

    res = {}
    for vs, n in zip(val_sets, val_sizes):
        for nq in p.n_test.unique():
            q_logits = np.array(p[p.n_test == nq].logits.tolist())
            k, v = ks_test_batch(vs, q_logits)
            assert k == (n, nq)
            res[k] = v

    with open('bbsd_mnist_no_shift.json', 'w') as f:
        json.dump({f'{k}': v for k,v in res.items()}, f)


if __name__ == '__main__':
    run_all_ks_tests()
