import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    A simple CNN model for image classification
    (1x28x28)
    """

    def __init__(self) -> None:
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)

        # Pooling layer
        self.pool = nn.MaxPool2d(2,2)

        # Fully connected layers
        # After two poolings: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 *7 * 7, 128)
        # Dropout after fc1, reduces overfitting
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Data flow over the network.
        """
        # Convolution -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Convolution -> BatchNorm -> ReLU -> Pooling
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = F.relu(x)
        # x = self.pool(x)
        #
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = F.relu(x)
        # x = self.pool(x)


        # Flatten tensors (batch_size, features)
        x = x.view(x.size(0), -1)

        # Layers fully connected
        x = F.relu(self.fc1(x))
        # x = self.dropout(x) # active only during training
        x = self.fc2(x)

        return x