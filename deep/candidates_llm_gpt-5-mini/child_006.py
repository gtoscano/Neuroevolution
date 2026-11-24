"""mini-AlexNet variant: introduce residual (skip) connections across convolutional pairs

One substantive change from parent: pairs of 3x3 convolutions are wrapped
into Residual blocks with identity (or 1x1 conv) shortcuts. BatchNorm stays
after each Conv and before ReLU; global average pooling head retained.
"""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a residual skip. If channels differ, use 1x1 conv on the skip."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = None
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.skip is not None:
            identity = self.skip(identity)
        out += identity
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # initial conv (5x5) like parent
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Residual block expanding 32 -> 64, then downsample
        self.res1 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
        )

        # Two residual blocks at 64 channels (keeps channel count)
        self.res2 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        # final conv to reduce channels to 32, then pool
        self.tail = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Global average pooling head -> 10 logits
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.tail(x)
        x = self.classifier(x)
        return x


def build_model():
    return Net()
