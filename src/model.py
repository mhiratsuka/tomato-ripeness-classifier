"""
This file creates a simple CNN model.

The model takes a tomato image and predicts:
- green
- ripe
- turning
"""

import torch
import torch.nn as nn

# Memo:
# conv1 = find basic features in the image
# conv2 = find more complex features
# ReLU = help the model learn non-linear patterns
# pool = reduce the size of the feature maps
# fc1 = combine the extracted features
# fc2 = output the three classes

class TomatoCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(3, 16, 3)

        # Second convolution layer
        self.conv2 = nn.Conv2d(16, 32, 3)

        # ReLU activation
        self.relu = nn.ReLU()

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 30 * 30, 64)
        self.fc2 = nn.Linear(64, 3)


    def forward(self, x):

        # First convolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Second convolution
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Change the image data into one long list of numbers
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)

        # Final prediction
        x = self.fc2(x)

        return x
