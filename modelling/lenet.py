import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    The most basic version of LeNet: https://en.wikipedia.org/wiki/LeNet
    """

    def __init__(self, in_channels=3, out_features=10):
        super(LeNet, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((28, 28))
        self.conv1 = nn.Conv2d(in_channels, 16, (5, 5), groups=1)
        self.conv2 = nn.Conv2d(16, 32, (5, 5), groups=1)

        self.mlp = nn.Linear(512, 256)
        self.fc = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.pool(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.mlp(x))
        x = self.fc(x)
        return x
