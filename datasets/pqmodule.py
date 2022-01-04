from copy import deepcopy

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

from utils.generic import vprint
from detectron.rejectronstep import RejectronStep
from datasets.subset import Subsetable


class PQModule(pl.LightningDataModule):
    def __init__(
            self,
            p: Dataset,
            p_prime: Dataset,
            q: Subsetable,
            batch_size=512,
            num_workers=64,
            verbose=True,
            drop_last=False,
    ):
        super().__init__()
        self.p = p
        self.q_base_length = len(q)
        self.q = Subsetable(q)
        self.p_prime = p_prime
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.print_fn = vprint(verbose=verbose)
        self.drop_last = drop_last

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader(ConcatDataset([self.p, self.q]))

    def p_base_dataloader(self) -> DataLoader:
        return self.__dataloader(self.p)

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader(self.p_prime)

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader(self.q)

    def __dataloader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            # pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )

    @property
    def n_train(self):
        return len(self.p)

    @property
    def n_test(self):
        return len(self.q)

    def refine(self, rs: RejectronStep):
        self.print_fn(f'{"-" * 60}\nRefining Q')
        # refines dataset q to only contains examples accepted by the rejectron step
        q_preds = []
        with torch.no_grad():
            for batch in tqdm(
                    DataLoader(self.q, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)):
                x, _ = batch
                y = rs.selective_classify(x)
                q_preds.append(y.cpu())

        q_preds = torch.cat(q_preds)
        q_indices = [i for i, p in enumerate(q_preds) if p != -1]
        lq = len(self.q)
        self.q.refine_dataset(q_indices)
        # with open('synthetic_data/rectangles/test_tmp.json', 'w') as f:
        #     json.dump([x[0].tolist() for x in self.q], f)

        self.print_fn(f'|Q|_old = {lq}, |Q|_new = {len(self.q)}\n{"-" * 60}')
