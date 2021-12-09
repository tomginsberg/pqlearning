import torch.nn as nn


class ImageMLP(nn.Module):
    """
    2 layer mlp + linear classifier for images
    """
    def __init__(self, in_channels=3, out_features=10, downsample_size=28):
        super(ImageMLP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((downsample_size, downsample_size))
        self.mlp = nn.Sequential(nn.Flatten(),
                                 nn.Linear(downsample_size * downsample_size * in_channels, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU())
        self.fc = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.pool(x)
        x = self.mlp(x)
        x = self.fc(x)
        return x


class ImageLinear(nn.Module):
    """
    Linear classifier for images
    """
    def __init__(self, in_channels=3, out_features=10, downsample_size=28):
        super(ImageLinear, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((downsample_size, downsample_size))
        self.fc = nn.Linear(downsample_size * downsample_size * in_channels, out_features)

    def forward(self, x):
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
