"""mini-AlexNet variant: introduce residual (skip) connections across convolutional blocks.

One substantive change from parent: replace some stacked conv layers with small residual blocks
(conv-BN-ReLU-conv-BN + skip connection). This should improve gradient flow and help the
network learn faster in the limited 2-epoch training regime.
"""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Basic 2-conv residual block. Adds a 1x1 conv skip when channels change."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            # match channels for the skip connection
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # initial conv stage (same as parent)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
        )

        # Residual block increasing channels 32 -> 64
        self.res1 = ResidualBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2)  # 14 -> 7

        # Another residual block keeping 64 channels
        self.res2 = ResidualBlock(64, 64)

        # Reduce channels before final pooling (similar to parent)
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7 -> 3
        )

        # Global average pooling head followed by a single linear layer.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # output: (batch, 32, 1, 1)
            nn.Flatten(),                  # output: (batch, 32)
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.conv_reduce(x)
        x = self.classifier(x)
        return x


def build_model():
    return Net()
