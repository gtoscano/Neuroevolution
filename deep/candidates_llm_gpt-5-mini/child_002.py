"""mini-AlexNet with global average pooling head + BatchNorm after each Conv

One substantive change from parent: BatchNorm layers inserted after each Conv2d
and before ReLU. This should stabilize and accelerate early training.
"""
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
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
