from modelling.image_model import ImageModel, CNNModule
from modelling.mlp import MLP

CHECKPOINT_DIR = '/voyager/projects/tomginsberg/pqlearning/checkpoints'


def resnet18_trained_on_mnist():
    return CNNModule.load_from_checkpoint(CHECKPOINT_DIR + '/mnist_baseline/epoch=6-step=3282.ckpt')
def resnet18_trained_on_cifar10():
    return ImageModel.load_from_checkpoint(CHECKPOINT_DIR + '/cifar/cifar10_resnet18/epoch=197-step=77417.ckpt')
def resnet18_trained_on_fashion_mnist():
    return CNNModule.load_from_checkpoint(CHECKPOINT_DIR + 'fashion_mnist_baseline_resnet/epoch=11-step=5627.ckpt')
def lenet_trained_on_mnist():
    return CNNModule.load_from_checkpoint(CHECKPOINT_DIR + '/mnist_baseline_lenet/epoch=24-step=9774.ckpt')
def lenet_trained_on_fashion_mnist():
    return CNNModule.load_from_checkpoint(CHECKPOINT_DIR + '/fashion_mnist_baseline_lenet/epoch=10-step=5158.ckpt')
def resnet18_trained_on_coil_p1():
    return CNNModule.load_from_checkpoint(CHECKPOINT_DIR + '/coil_baseline/epoch=15-step=127.ckpt')
def lenet_trained_on_coil_p1():
    return CNNModule.load_from_checkpoint(CHECKPOINT_DIR + '/coil_baseline_lenet/epoch=35-step=287.ckpt')
def mlp_trained_on_coil_p1():
    return CNNModule.load_from_checkpoint(CHECKPOINT_DIR + '/coil_baseline_mlp/epoch=46-step=375.ckpt')
def linear_trained_on_coil_p1():
    return CNNModule.load_from_checkpoint(CHECKPOINT_DIR + '/coil_baseline_linear/epoch=42-step=343.ckpt')

def mlp_large_trained_on_rectangles() -> MLP:
    return MLP.load_from_checkpoint(CHECKPOINT_DIR + '/rectangles/rectangles_mlp_large-epoch=9-val_acc=0.98.ckpt')
