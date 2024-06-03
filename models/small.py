import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import Builder
from args import args

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        builder = Builder()
        self.convs = nn.Sequential(
            builder.conv3x3(1, 32, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(32, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.linear = nn.Sequential(
            builder.conv1x1(12544, 128),
            nn.ReLU(),
            builder.conv1x1(128, 10),
        )
    def forward(self, x):
        out = x.view(x.size(0), 1,  28, 28)
        out = self.convs(out)
        out = out.view(out.size(0), 64* 14 * 14, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        builder = Builder()
        self.convs = nn.Sequential(
            builder.conv3x3(1, 64, first_layer=True),  # Change input channels to 1
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # Max pooling after first conv layer

            builder.conv3x3(64, 128),  # Second conv layer
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # Max pooling after second conv layer
        )

        self.linear = nn.Sequential(
            builder.conv1x1(128 * 7 * 7, 256),  # Adjust dimensions accordingly
            nn.ReLU(),
            builder.conv1x1(256, 10),  # Fully connected layer 2 (output layer)
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 128 * 7 * 7, 1, 1)  # Flatten the output for the fully connected layers
        out = self.linear(out)
        return out.squeeze()

class Conv8(nn.Module):
    def __init__(self):
        super(Conv8, self).__init__()
        builder = Builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(128, 256),
            nn.ReLU(),
            builder.conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(256, 512),
            nn.ReLU(),
            builder.conv3x3(512, 512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 2 * 2, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.squeeze()