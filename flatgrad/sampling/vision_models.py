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
        Conv(1->32, 3x3) -> BN -> ReLU -> MaxPool(2x2)
        Conv(32->64, 3x3) -> BN -> ReLU -> MaxPool(2x2)
        Flatten -> FC(64*7*7 -> 128) -> BN -> ReLU -> Dropout(configurable)
        FC(128 -> 10)
    
    Expected accuracy: ~98-99% on MNIST
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 2 pooling layers: 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        
        # Store dropout rate for inspection
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Input: [B, 1, 28, 28]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 32, 14, 14]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)  # [B, 64*7*7]
        x = F.relu(self.bn3(self.fc1(x)))  # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)  # [B, 10]
        return x


class MNISTMLP(nn.Module):
    """
    Very large MLP for MNIST - designed to strongly overfit without regularization.
    
    Uses very large hidden layers to create extreme model capacity. Without
    regularization, this model will overfit heavily on small-to-moderate datasets,
    creating strong room for regularization methods to show their benefits.
    
    Architecture:
        Flatten(784) -> FC(784 -> 2048) -> ReLU -> Dropout(configurable)
        FC(2048 -> 1024) -> ReLU -> Dropout(configurable)
        FC(1024 -> 512) -> ReLU
        FC(512 -> 10)
    
    Parameters: ~2.6M (very high capacity for 28x28 images)
    Expected: Strong overfitting on 2500 samples without regularization
    """
    
    def __init__(self, dropout_rate: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(784, 2048)  # Doubled from 1024
        self.fc2 = nn.Linear(2048, 1024)  # Doubled from 512
        self.fc3 = nn.Linear(1024, 512)   # Doubled from 256
        self.fc4 = nn.Linear(512, 10)
        
        # Store dropout rate for inspection
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Input: [B, 1, 28, 28]
        x = x.view(x.size(0), -1)  # [B, 784]
        x = F.relu(self.fc1(x))  # [B, 2048]
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # [B, 1024]
        x = self.dropout(x)
        x = F.relu(self.fc3(x))  # [B, 512]
        x = self.fc4(x)  # [B, 10]
        return x


class CIFAR10MLP(nn.Module):
    """
    Simple MLP for CIFAR-10 (32x32 RGB images).
    
    Uses flattened input (3072 features) instead of convolutions.
    This reduces model capacity significantly compared to CNNs, making it:
    - Harder to achieve 100% train accuracy (prevents memorization)
    - Faster to train (fewer parameters)
    - Still capable of learning meaningful patterns
    
    Architecture:
        Flatten(3072) -> FC(3072 -> 256) -> ReLU -> Dropout
        FC(256 -> 128) -> ReLU -> Dropout
        FC(128 -> 10)
    
    Parameters: ~800k (vs CNN's ~1.5M)
    Expected accuracy: ~50-60% on CIFAR-10 (much lower than CNN's 80-85%)
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 256)  # Reduced from 512
        self.fc2 = nn.Linear(256, 128)  # Reduced from 256
        self.fc3 = nn.Linear(128, 10)
        
        # Store dropout rate for inspection
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Input: [B, 3, 32, 32]
        x = x.view(x.size(0), -1)  # [B, 3072]
        x = F.relu(self.fc1(x))  # [B, 256]
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # [B, 128]
        x = self.dropout(x)
        x = self.fc3(x)  # [B, 10]
        return x


class CIFAR10ConvNet(nn.Module):
    """
    Simple convolutional network for CIFAR-10 (32x32 RGB images).
    
    Architecture:
        Conv(3->32, 3x3) -> BN -> ReLU -> Conv(32->32, 3x3) -> BN -> ReLU -> MaxPool(2x2)
        Conv(32->64, 3x3) -> BN -> ReLU -> Conv(64->64, 3x3) -> BN -> ReLU -> MaxPool(2x2)
        Flatten -> FC(64*8*8 -> 256) -> BN -> ReLU -> Dropout(configurable)
        FC(256 -> 10)
    
    Expected accuracy: ~80-85% on CIFAR-10 (with proper training)
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 2 pooling layers: 32 -> 16 -> 8
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
        
        # Store dropout rate for inspection
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Input: [B, 3, 32, 32]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 32, 32, 32]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 32, 16, 16]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 64, 16, 16]
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # [B, 64, 8, 8]
        x = x.view(x.size(0), -1)  # [B, 64*8*8]
        x = F.relu(self.bn5(self.fc1(x)))  # [B, 256]
        x = self.dropout(x)
        x = self.fc2(x)  # [B, 10]
        return x


def create_vision_model(dataset: str, dropout_rate: float = 0.5) -> nn.Module:
    """
    Factory function to create vision models.
    
    Args:
        dataset: 'mnist', 'mnist_convnet', 'mnist_mlp', 'cifar10', 'cifar10_convnet', or 'cifar10_mlp'
        dropout_rate: Dropout probability (default: 0.5)
    
    Returns:
        Neural network model
    
    Example:
        >>> model = create_vision_model('mnist_mlp', dropout_rate=0.0)  # Baseline with no dropout
        >>> model = create_vision_model('mnist_mlp', dropout_rate=0.5)  # With dropout regularization
    """
    dataset = dataset.lower()
    
    # Handle MNIST models
    if dataset == 'mnist_convnet':
        return MNISTConvNet(dropout_rate=dropout_rate)
    elif dataset in ('mnist', 'mnist_mlp'):
        return MNISTMLP(dropout_rate=dropout_rate)
    elif dataset in ('cifar10', 'cifar10_convnet'):
        return CIFAR10ConvNet(dropout_rate=dropout_rate)
    elif dataset == 'cifar10_mlp':
        return CIFAR10MLP(dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: 'mnist', 'mnist_convnet', 'mnist_mlp', 'cifar10', 'cifar10_convnet', 'cifar10_mlp'")


# Alias for backward compatibility
get_vision_model = create_vision_model
