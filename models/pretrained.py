from .cnn import CNN


def resnet18_trained_on_mnist():
    return CNN.load_from_checkpoint('~/pqlearning/checkpoints/mnist_baseline/epoch=6-step=3282.ckpt')
