from collections import Callable
from copy import deepcopy
from os import path
from shutil import rmtree
from typing import Optional, List, Dict

import pandas as pd
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from datasets.pqmodule import PQModule
from detectron import RejectronStep
import traceback


def rejectron_trainer(save_directory: str, run_name: str = None,
                      gpus=1, max_epochs=50, monitor: str = 'p_q_score',
                      patience: int = 15, dryrun=False, delete_previous=True, **kwargs):
    if not dryrun:
        assert run_name is not None, "Must provide a run name if dryrun is False"

    if path.isdir(f'checkpoints/{save_directory}'):
        if delete_previous:
            print(f'Deleting previous run at {save_directory}')
            rmtree(f'checkpoints/{save_directory}')

    def f(iteration: int):
        return pl.Trainer(
            gpus=gpus, auto_select_gpus=True, max_epochs=max_epochs, log_every_n_steps=1, num_sanity_val_steps=0,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    f'checkpoints/{save_directory}',
                    filename=f'c{iteration}_' + '{epoch}_{p_q_score:.5f}',
                    save_top_k=1,
                    monitor=monitor,
                    mode='max',
                    verbose=True,
                ),
                pl.callbacks.EarlyStopping(monitor=monitor, patience=patience, mode='max', verbose=True),
            ],
            logger=False if dryrun else WandbLogger(project="pqlearning", offline=dryrun,
                                                    name=f'{run_name}_{iteration}'), **kwargs
        )

    return f


def benchmark(h: pl.LightningModule, pq: PQModule) -> pd.DataFrame:
    df = pd.DataFrame()
    print('Benchmarking h')
    t = pl.Trainer(gpus=1)
    tr = t.validate(h, pq.p_base_dataloader())[0]['val_acc']
    va = t.validate(h, pq.val_dataloader())[0]['val_acc']
    te = t.validate(h, pq.test_dataloader())[0]['val_acc']
    df = df.append(dict(Step=0,
                        train_agree=1,
                        test_reject=0,
                        train_acc=tr,
                        test_acc=te,
                        p_q_score=10,
                        val_agree=1,
                        val_acc=va), ignore_index=True)
    print(df)
    return df


def train_rejectors(pq: PQModule, h: pl.LightningModule, trainer: Callable[[int], pl.Trainer],
                    num_rejectors: int = 16,
                    logfile: Optional[str] = None,
                    patience: int = 5):
    size_of_q = len(pq.q)
    count = 0

    df = benchmark(h, pq)


    for i in range(1, num_rejectors + 1):
        c_step = RejectronStep(h, n_train=pq.n_train, n_test=pq.n_test)
        try:
            trainer(i).fit(model=c_step, datamodule=pq)
            wandb.finish()
        except KeyboardInterrupt:
            print('Keyboard interrupt')
            raise KeyboardInterrupt()
        except Exception:
            # print stack trace
            traceback.print_exc()
            print(f'Failed on step {i}')
            break

        lq_1 = len(pq.q)
        pq.refine(rs=c_step)
        lq_2 = len(pq.q)

        if lq_2 < lq_1:
            size_of_q = lq_2
        else:
            count += 1

        df = format_results(results=c_step.get_results(), df=df, i=i, pq=pq, size_of_q=size_of_q)

        if logfile is not None:
            df.to_csv(logfile + '.csv', index=False)

        if count > patience:
            print(f'Rejectors have not improved for {patience} rounds. Stopping training.')
            break

        if lq_2 == 0:
            print(f'100% rejection. Stopping training.')
            break

    return df


def format_results(results: List[Dict[str, float]], df: pd.DataFrame, i: int, pq: PQModule, size_of_q: int):
    """
    Get the results of the current rejector and add them to the dataframe.
    Args:
        results: current results
        df: dataframe
        i: rejector index
        pq: datasets
        size_of_q: how mich q data is left

    Returns: dataframe with results appended

    """
    train_test, val = results
    # update q_disagree
    q_disagree = 1 - size_of_q / pq.q_base_length
    p_q_score = int(train_test['p_q_score']) + q_disagree
    res = ({'Step': i} | train_test | val)
    res.update(p_q_score=p_q_score, test_reject=q_disagree)
    df = df.append(res, ignore_index=True)
    print(df)
    return df
