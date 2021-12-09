from .cnn import CNN

CHECKPOINT_DIR = '/voyager/projects/tomginsberg/pqlearning/checkpoints'


def resnet18_trained_on_mnist():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/mnist_baseline/epoch=6-step=3282.ckpt')


def resnet18_trained_on_cifar10():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/cifar10_baseline_resnet18/epoch=193-step=75853.ckpt')


def resnet18_trained_on_fashion_mnist():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + 'fashion_mnist_baseline_resnet/epoch=11-step=5627.ckpt')


def lenet_trained_on_mnist():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/mnist_baseline_lenet/epoch=24-step=9774.ckpt')


def lenet_trained_on_fashion_mnist():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/fashion_mnist_baseline_lenet/epoch=10-step=5158.ckpt')


def resnet18_trained_on_coil_p1():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/coil_baseline/epoch=15-step=127.ckpt')


def lenet_trained_on_coil_p1():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/coil_baseline_lenet/epoch=35-step=287.ckpt')


def mlp_trained_on_coil_p1():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/coil_baseline_mlp/epoch=46-step=375.ckpt')


def linear_trained_on_coil_p1():
    return CNN.load_from_checkpoint(CHECKPOINT_DIR + '/coil_baseline_linear/epoch=42-step=343.ckpt')
