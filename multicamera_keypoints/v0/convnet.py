import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """Convnet used in the segmentation (CENTERNET) step.
    """
    def __init__(self, initial_channels=32):
        super(ConvNet, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(1, initial_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(initial_channels)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(
            initial_channels, initial_channels * 2, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(initial_channels * 2)

        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(
            initial_channels * 2,
            initial_channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(initial_channels * 4)

        # Fourth Convolutional Layer
        self.conv4 = nn.Conv2d(
            initial_channels * 4,
            initial_channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn4 = nn.BatchNorm2d(initial_channels * 4)

        # MaxPooling Layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Final Convolutional Layer to generate 1-channel output
        self.conv_final = nn.Conv2d(initial_channels * 4, 1, kernel_size=1, stride=1)

    def forward(self, x):
        # First Conv + ReLU + MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Second Conv + ReLU + MaxPool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Third Conv + ReLU + MaxPool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Fourth Conv + ReLU + MaxPool
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Final Conv
        x = self.conv_final(x)
        x = F.sigmoid(x)
        return x
