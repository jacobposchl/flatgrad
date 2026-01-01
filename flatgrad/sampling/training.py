"""
Training utilities for neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Dict, List
import time
import numpy as np


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        logits: Model logits [N, C]
        labels: Ground truth labels [N]
        n_bins: Number of bins for calibration (default: 10)
    
    Returns:
        ECE score (lower is better)
    """
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, dim=1)
    accuracies = predictions.eq(labels)
    
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


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
        Dictionary with 'loss', 'accuracy', and optionally 'reg_loss' and 'main_loss' for the epoch
    """
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_reg_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)
        main_loss = loss_fn(logits, labels)
        
        # Add regularization if provided (applied per batch)
        reg_loss = torch.tensor(0.0, device=device)
        if regularizer_fn is not None:
            reg_loss = regularizer_fn(model, inputs, labels, loss_fn)
            loss = main_loss + reg_loss
        else:
            loss = main_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_main_loss += main_loss.item() * batch_size
        total_reg_loss += reg_loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
    
    result = {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'main_loss': total_main_loss / total_samples,
        'reg_loss': total_reg_loss / total_samples
    }
    
    return result


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: torch.device,
    compute_calibration: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Neural network
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device to evaluate on
        compute_calibration: Whether to compute ECE (default: False)
    
    Returns:
        Dictionary with 'loss', 'accuracy', and optionally 'ece'
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item() * inputs.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += inputs.size(0)
            
            if compute_calibration:
                all_logits.append(logits)
                all_labels.append(labels)
    
    result = {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }
    
    if compute_calibration and len(all_logits) > 0:
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        result['ece'] = compute_ece(all_logits, all_labels)
    
    return result


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


class RegTracker:
    """
    Tracks regularization magnitude during training.
    
    Example:
        >>> tracker = RegTracker()
        >>> tracker.record(epoch=0, reg_loss=0.01, main_loss=0.5)
        >>> tracker.record(epoch=1, reg_loss=0.008, main_loss=0.3)
        >>> tracker.get_history()
    """
    
    def __init__(self):
        self.epochs = []
        self.reg_losses = []
        self.main_losses = []
        self.reg_ratios = []  # reg_loss / main_loss
        self.timestamps = []
    
    def record(self, epoch: int, reg_loss: float, main_loss: float):
        """Record regularization statistics for an epoch."""
        self.epochs.append(epoch)
        self.reg_losses.append(reg_loss)
        self.main_losses.append(main_loss)
        ratio = reg_loss / main_loss if main_loss > 0 else 0.0
        self.reg_ratios.append(ratio)
        self.timestamps.append(time.time())
    
    def get_history(self) -> Dict[str, List]:
        """Get complete tracking history."""
        return {
            'epochs': self.epochs,
            'reg_losses': self.reg_losses,
            'main_losses': self.main_losses,
            'reg_ratios': self.reg_ratios,
            'timestamps': self.timestamps
        }
    
    def get_latest(self) -> Optional[Dict[str, float]]:
        """Get most recent regularization statistics."""
        if len(self.epochs) == 0:
            return None
        return {
            'epoch': self.epochs[-1],
            'reg_loss': self.reg_losses[-1],
            'main_loss': self.main_losses[-1],
            'reg_ratio': self.reg_ratios[-1]
        }
    
    def print_summary(self):
        """Print summary of regularization evolution."""
        if len(self.epochs) == 0:
            print("No regularization measurements recorded.")
            return
        
        print("\n" + "="*60)
        print("Regularization Magnitude Summary")
        print("="*60)
        print(f"{'Epoch':<10} {'Reg Loss':<15} {'Main Loss':<15} {'Ratio':<15}")
        print("-"*60)
        
        for epoch, reg, main, ratio in zip(self.epochs, self.reg_losses, self.main_losses, self.reg_ratios):
            print(f"{epoch:<10} {reg:<15.6f} {main:<15.6f} {ratio:<15.6f}")
        
        # Summary statistics
        if len(self.reg_losses) > 0:
            print("-"*60)
            print(f"Mean Reg Loss: {np.mean(self.reg_losses):.6f}")
            print(f"Mean Reg Ratio: {np.mean(self.reg_ratios):.6f}")
            print(f"Max Reg Ratio: {np.max(self.reg_ratios):.6f}")
        print("="*60 + "\n")
