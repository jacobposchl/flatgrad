"""
Neural network architectures for vision tasks (MNIST, CIFAR-10).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTConvNet(nn.Module):
    """
    Simple convolutional network for MNIST (28x28 grayscale images).
    
    Architecture:
        Conv(1->32, 3x3) -> ReLU -> MaxPool(2x2)
        Conv(32->64, 3x3) -> ReLU -> MaxPool(2x2)
        Flatten -> FC(64*7*7 -> 128) -> ReLU -> Dropout(0.5)
        FC(128 -> 10)
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 2 pooling layers: 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Input: [B, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)  # [B, 64*7*7]
        x = F.relu(self.fc1(x))  # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)  # [B, 10]
        return x


class CIFAR10ConvNet(nn.Module):
    """
    Simple convolutional network for CIFAR-10 (32x32 RGB images).
    
    Architecture:
        Conv(3->32, 3x3) -> ReLU -> Conv(32->32, 3x3) -> ReLU -> MaxPool(2x2)
        Conv(32->64, 3x3) -> ReLU -> Conv(64->64, 3x3) -> ReLU -> MaxPool(2x2)
        Flatten -> FC(64*8*8 -> 256) -> ReLU -> Dropout(0.5)
        FC(256 -> 10)
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 2 pooling layers: 32 -> 16 -> 8
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Input: [B, 3, 32, 32]
        x = F.relu(self.conv1(x))  # [B, 32, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 16, 16]
        x = F.relu(self.conv3(x))  # [B, 64, 16, 16]
        x = self.pool(F.relu(self.conv4(x)))  # [B, 64, 8, 8]
        x = x.view(x.size(0), -1)  # [B, 64*8*8]
        x = F.relu(self.fc1(x))  # [B, 256]
        x = self.dropout(x)
        x = self.fc2(x)  # [B, 10]
        return x


def create_vision_model(dataset: str, dropout_rate: float = 0.5) -> nn.Module:
    """
    Factory function to create vision models.
    
    Args:
        dataset: 'mnist' or 'cifar10'
        dropout_rate: Dropout probability (default: 0.5)
    
    Returns:
        Neural network model
    
    Example:
        >>> model = create_vision_model('mnist')
        >>> model = create_vision_model('cifar10', dropout_rate=0.3)
    """
    dataset = dataset.lower()
    
    if dataset == 'mnist':
        return MNISTConvNet(dropout_rate=dropout_rate)
    elif dataset == 'cifar10':
        return CIFAR10ConvNet(dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: 'mnist', 'cifar10'")
