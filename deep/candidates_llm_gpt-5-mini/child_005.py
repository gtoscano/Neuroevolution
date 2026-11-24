"""mini-AlexNet variant using depthwise-separable convolutions in the later stages

One substantive change from parent: the two 3x3 Conv2d layers after the second
pool are replaced with depthwise-separable convolutions (depthwise 3x3 followed
by 1x1 pointwise). BatchNorm is kept after each pointwise conv, before ReLU,
matching the parent's BN placement. This is intended to increase parameter
efficiency and improve early training on the small training budget.
"""
import torch
from torch import nn


class SeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise 3x3 then pointwise 1x1.
    BatchNorm applied after pointwise conv, then ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # depthwise conv: groups=in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        # pointwise conv to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # Replace the later standard 3x3 convs with depthwise-separable convs
            SeparableConv(64, 64, kernel_size=3, padding=1),
            SeparableConv(64, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
        )
        # Global average pooling head followed by a single linear layer.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # output: (batch, 32, 1, 1)
            nn.Flatten(),                  # output: (batch, 32)
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model():
    return Net()
