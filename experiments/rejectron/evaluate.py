import pytorch_lightning as pl
from glob import glob
import torch
import os
from tqdm import tqdm

from datasets import MnistDataModule
from modelling import CNN
from modelling.pretrained import resnet18_trained_on_mnist
from experiments.rejectron.rejectron import RejectronClassifier

# os.chdir(f'{os.environ["HOME"]}/pqlearning')


checkpoints = sorted(
    glob('checkpoints/rejectron_mnist/c*/*ckpt'),
    # sort by model num in "checkpoints/rejectron_mnist/c_(model num)"
    key=lambda x: int(x.split('/')[2].split('_')[-1])
)
rejectors = [resnet18_trained_on_mnist().cuda()] + \
            [CNN.load_from_checkpoint(checkpoint_path=checkpoint).cuda() for checkpoint in checkpoints]

mnist = MnistDataModule(shift_transform_type='natural', test_transform_rate=.5, rotation=20, crop=.3, distortion=.1,
                        batch_size=256)

y_true = []
y_pred = []
for x, y in tqdm(mnist.test_dataloader()):
    y_true.append(y)
    y_pred.append(torch.stack([f(x).argmax(1).cpu() for f in rejectors]))

y_true = torch.cat(y_true)
y_pred = torch.cat(y_pred)
