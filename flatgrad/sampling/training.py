"""
Training utilities for neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Dict, List
import time


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    regularizer_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function (e.g., F.cross_entropy)
        device: Device to train on
        regularizer_fn: Optional regularization function taking (model, inputs, labels, loss_fn)
    
    Returns:
        Dictionary with 'loss' and 'accuracy' for the epoch
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        
        # Add regularization if provided
        if regularizer_fn is not None:
            reg_loss = regularizer_fn(model, inputs, labels, loss_fn)
            loss = loss + reg_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += inputs.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Neural network
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device to evaluate on
    
    Returns:
        Dictionary with 'loss' and 'accuracy'
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item() * inputs.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += inputs.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }


class LambdaTracker:
    """
    Tracks lambda evolution during training.
    
    Example:
        >>> tracker = LambdaTracker()
        >>> tracker.record(epoch=0, lambda_mean=0.45, lambda_std=0.12)
        >>> tracker.record(epoch=1, lambda_mean=0.42, lambda_std=0.10)
        >>> tracker.get_history()
    """
    
    def __init__(self):
        self.epochs = []
        self.lambda_means = []
        self.lambda_stds = []
        self.timestamps = []
    
    def record(self, epoch: int, lambda_mean: float, lambda_std: float):
        """Record lambda statistics for an epoch."""
        self.epochs.append(epoch)
        self.lambda_means.append(lambda_mean)
        self.lambda_stds.append(lambda_std)
        self.timestamps.append(time.time())
    
    def get_history(self) -> Dict[str, List]:
        """Get complete tracking history."""
        return {
            'epochs': self.epochs,
            'lambda_means': self.lambda_means,
            'lambda_stds': self.lambda_stds,
            'timestamps': self.timestamps
        }
    
    def get_latest(self) -> Optional[Dict[str, float]]:
        """Get most recent lambda statistics."""
        if len(self.epochs) == 0:
            return None
        return {
            'epoch': self.epochs[-1],
            'lambda_mean': self.lambda_means[-1],
            'lambda_std': self.lambda_stds[-1]
        }
    
    def print_summary(self):
        """Print summary of lambda evolution."""
        if len(self.epochs) == 0:
            print("No lambda measurements recorded.")
            return
        
        print("\n" + "="*60)
        print("Lambda Evolution Summary")
        print("="*60)
        print(f"{'Epoch':<10} {'λ Mean':<15} {'λ Std':<15}")
        print("-"*60)
        
        for epoch, mean, std in zip(self.epochs, self.lambda_means, self.lambda_stds):
            print(f"{epoch:<10} {mean:<15.6f} {std:<15.6f}")
        
        # Summary statistics
        initial_lambda = self.lambda_means[0]
        final_lambda = self.lambda_means[-1]
        lambda_change = final_lambda - initial_lambda
        lambda_change_pct = (lambda_change / initial_lambda) * 100 if initial_lambda != 0 else 0
        
        print("-"*60)
        print(f"Initial λ: {initial_lambda:.6f}")
        print(f"Final λ:   {final_lambda:.6f}")
        print(f"Change:    {lambda_change:.6f} ({lambda_change_pct:+.2f}%)")
        print("="*60 + "\n")
