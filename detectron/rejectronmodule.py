import pytorch_lightning as pl
import torch
from detectron.rejectronstep import RejectronStep


class RejectronModule(pl.LightningModule):
    def __init__(self, h=pl.LightningModule, **kwargs):
        super().__init__(**kwargs)
        self.C = []
        self.h = h

    def add_new_c(self, c_step: RejectronStep):
        self.C.append(c_step.get_c())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = torch.argmax(self.h(x), dim=1)

            for c in self.C:
                # mask out indices with -1 whenever a c model disagrees with h
                mask = torch.argmax(c(x), dim=1) != y
                y[mask] = -1

            return y
