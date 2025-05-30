import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_2d(nn.Module):
    def __init__(self, input_channels=1, num_classes=12):
        super(CNN_2d, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.1)
        )

        self.additional_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten_size = self._get_flatten_size(input_channels)

        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout_fc2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def _get_flatten_size(self, input_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 1292, 128)
            x = self.conv_block1(dummy)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.additional_pool(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.additional_pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        x = self.fc3(x)

        return x