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
    Tracks lambda evolution during training with full per-direction data retention.
    
    Example:
        >>> tracker = LambdaTracker(store_full_data=True)
        >>> tracker.record(epoch=0, lambda_mean=0.45, lambda_std=0.12, 
        ...                lambda_values=[0.4, 0.5, 0.45], derivatives_per_dir=[[...], [...], [...]])
        >>> tracker.get_history()
    """
    
    def __init__(self, store_full_data: bool = False):
        """
        Args:
            store_full_data: If True, stores all per-direction lambda values and derivatives.
                            If False, only stores aggregate statistics (mean/std).
        """
        self.epochs = []
        self.lambda_means = []
        self.lambda_stds = []
        self.timestamps = []
        self.store_full_data = store_full_data
        
        # Full data storage (only used if store_full_data=True)
        self.lambda_values_per_epoch = []  # List of lists: [[dir1_lambda, dir2_lambda, ...], ...]
        self.derivatives_per_epoch = []    # List of lists of lists: [[[dir1_order1, dir1_order2, ...], [dir2_order1, ...]], ...]
    
    def record(self, epoch: int, lambda_mean: float, lambda_std: float, 
               lambda_values: list = None, derivatives_per_dir: list = None):
        """
        Record lambda statistics for an epoch.
        
        Args:
            epoch: Training epoch
            lambda_mean: Mean lambda across directions
            lambda_std: Standard deviation of lambda
            lambda_values: List of lambda values (one per direction). Only stored if store_full_data=True.
            derivatives_per_dir: List of derivative lists (one per direction). Only stored if store_full_data=True.
        """
        self.epochs.append(epoch)
        self.lambda_means.append(lambda_mean)
        self.lambda_stds.append(lambda_std)
        self.timestamps.append(time.time())
        
        if self.store_full_data:
            if lambda_values is not None:
                self.lambda_values_per_epoch.append(lambda_values)
            else:
                self.lambda_values_per_epoch.append([])
            
            if derivatives_per_dir is not None:
                self.derivatives_per_epoch.append(derivatives_per_dir)
            else:
                self.derivatives_per_epoch.append([])
    
    def get_history(self) -> Dict[str, List]:
        """Get complete tracking history."""
        history = {
            'epochs': self.epochs,
            'lambda_means': self.lambda_means,
            'lambda_stds': self.lambda_stds,
            'timestamps': self.timestamps
        }
        
        if self.store_full_data:
            history['lambda_values_per_epoch'] = self.lambda_values_per_epoch
            history['derivatives_per_epoch'] = self.derivatives_per_epoch
        
        return history
    
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


def adaptive_measurement_schedule(total_epochs: int) -> List[int]:
    """
    Generate adaptive lambda measurement schedule.
    
    Measures more frequently early in training when lambda changes rapidly,
    less frequently later when lambda has stabilized.
    
    Args:
        total_epochs: Total number of training epochs
    
    Returns:
        List of epoch numbers when lambda should be measured
    """
    measurement_epochs = []
    
    # Epochs 0-10: Every epoch (dense early tracking)
    for epoch in range(min(11, total_epochs + 1)):
        measurement_epochs.append(epoch)
    
    # Epochs 11-30: Every 2 epochs
    if total_epochs > 10:
        for epoch in range(11, min(31, total_epochs + 1), 2):
            if epoch not in measurement_epochs:
                measurement_epochs.append(epoch)
    
    # Epochs 31-50: Every 5 epochs
    if total_epochs > 30:
        for epoch in range(31, min(51, total_epochs + 1), 5):
            if epoch not in measurement_epochs:
                measurement_epochs.append(epoch)
    
    # Epochs 50+: Every 10 epochs
    if total_epochs > 50:
        for epoch in range(51, total_epochs + 1, 10):
            if epoch not in measurement_epochs:
                measurement_epochs.append(epoch)
    
    return sorted(measurement_epochs)


def save_lambda_data(tracker: LambdaTracker, output_path: str, 
                     train_accuracies: List[float] = None,
                     test_accuracies: List[float] = None,
                     train_losses: List[float] = None,
                     test_losses: List[float] = None):
    """
    Save lambda tracking data to .npz file for post-hoc analysis.
    
    Args:
        tracker: LambdaTracker with recorded data
        output_path: Path to save .npz file (should end in .npz)
        train_accuracies: List of train accuracies corresponding to measurement epochs
        test_accuracies: List of test accuracies corresponding to measurement epochs
        train_losses: List of train losses corresponding to measurement epochs
        test_losses: List of test losses corresponding to measurement epochs
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    history = tracker.get_history()
    
    # Prepare data dictionary
    data = {
        'epochs': np.array(history['epochs']),
        'lambda_means': np.array(history['lambda_means']),
        'lambda_stds': np.array(history['lambda_stds']),
        'timestamps': np.array(history['timestamps']),
    }
    
    # Add optional metrics if provided
    if train_accuracies is not None:
        data['train_accuracies'] = np.array(train_accuracies)
    if test_accuracies is not None:
        data['test_accuracies'] = np.array(test_accuracies)
    if train_losses is not None:
        data['train_losses'] = np.array(train_losses)
    if test_losses is not None:
        data['test_losses'] = np.array(test_losses)
    
    # Add full data if available
    if tracker.store_full_data:
        # Convert lists of lists to arrays
        # lambda_values_per_epoch: list of lists (variable length per epoch)
        # derivatives_per_epoch: list of lists of lists
        
        # Store as object arrays to handle variable-length data
        data['lambda_values_per_epoch'] = np.array(history['lambda_values_per_epoch'], dtype=object)
        data['derivatives_per_epoch'] = np.array(history['derivatives_per_epoch'], dtype=object)
    
    np.savez(output_path, **data)


def save_training_metrics(epochs: List[int], 
                          train_accuracies: List[float],
                          test_accuracies: List[float],
                          train_losses: List[float],
                          test_losses: List[float],
                          output_path: str,
                          eces: List[float] = None,
                          lambda_means: List[float] = None,
                          lambda_stds: List[float] = None):
    """
    Save epoch-by-epoch training metrics to CSV file.
    
    Args:
        epochs: List of epoch numbers
        train_accuracies: Training accuracies
        test_accuracies: Test accuracies
        train_losses: Training losses
        test_losses: Test losses
        output_path: Path to save CSV file
        eces: Expected calibration errors (optional)
        lambda_means: Lambda mean values (optional)
        lambda_stds: Lambda std values (optional)
    """
    import pandas as pd
    import os
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        'epoch': epochs,
        'train_accuracy': train_accuracies,
        'test_accuracy': test_accuracies,
        'train_loss': train_losses,
        'test_loss': test_losses,
        'generalization_gap': [train_acc - test_acc for train_acc, test_acc in zip(train_accuracies, test_accuracies)]
    }
    
    if eces is not None:
        data['ece'] = eces
    if lambda_means is not None:
        data['lambda_mean'] = lambda_means
    if lambda_stds is not None:
        data['lambda_std'] = lambda_stds
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
