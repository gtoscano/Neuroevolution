"""mini-AlexNet with global average pooling head"""
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Substantive change: replace large fully-connected classifier with
        # a global average pooling head followed by a single linear layer.
        # This greatly reduces parameters and encourages spatially-averaged
        # features for classification, which often helps with small-data fast learning.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # output: (batch, 32, 1, 1)
            nn.Flatten(),                  # output: (batch, 32)
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model():
    return Net()
