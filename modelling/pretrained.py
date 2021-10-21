from .cnn import CNN

CHECKPOINT_DIR = '/voyager/projects/tomginsberg/pqlearning/checkpoints'


def resnet18_trained_on_mnist():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/mnist_baseline/epoch=6-step=3282.ckpt')


def lenet_trained_on_mnist():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/mnist_baseline_lenet/epoch=7-step=3751.ckpt')


def resnet18_trained_on_coil_p1():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/coil_baseline/epoch=15-step=127.ckpt')


def lenet_trained_on_coil_p1():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/coil_baseline_lenet/epoch=35-step=287.ckpt')
