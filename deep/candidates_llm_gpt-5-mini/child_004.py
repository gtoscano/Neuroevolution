"""mini-AlexNet variant using Group Normalization instead of Batch Normalization.

One substantive change from parent: replace all nn.BatchNorm2d with nn.GroupNorm(num_groups=8, num_channels=...).
GroupNorm is more robust to small batch sizes and should help the model learn faster during the limited 2-epoch
training on a small subset of data.
"""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Basic 2-conv residual block. Adds a 1x1 conv skip when channels change.

    Uses GroupNorm instead of BatchNorm for more stable training with small batches.
    """

    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # GroupNorm: number of groups must divide channels
        g1 = groups if out_channels % groups == 0 else 1
        self.gn1 = nn.GroupNorm(g1, out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        g2 = groups if out_channels % groups == 0 else 1
        self.gn2 = nn.GroupNorm(g2, out_channels)
        if in_channels != out_channels:
            # match channels for the skip connection
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # initial conv stage (same as parent but with GroupNorm)
        # choose groups so they divide 32
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
        )

        # Residual block increasing channels 32 -> 64
        self.res1 = ResidualBlock(32, 64, groups=8)
        self.pool1 = nn.MaxPool2d(2)  # 14 -> 7

        # Another residual block keeping 64 channels
        self.res2 = ResidualBlock(64, 64, groups=8)

        # Reduce channels before final pooling (similar to parent)
        # use GroupNorm here as well
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
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
