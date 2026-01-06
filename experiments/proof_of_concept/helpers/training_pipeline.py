"""
Unified training pipeline for proof-of-concept experiments.

Handles model training, lambda measurement, checkpointing, and data saving.
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from flatgrad.sampling.training import (
    LambdaTracker, evaluate, compute_ece, 
    adaptive_measurement_schedule, save_lambda_data, save_training_metrics
)
from flatgrad.sampling.lambda_estimation import estimate_lambda_per_direction
from flatgrad.sampling.vision_models import get_vision_model
from .experiment_config import ExperimentConfig
from .regularization_methods import (
    get_optimizer_with_config, get_data_augmentation_transforms, 
    get_standard_transforms, LabelSmoothingLoss, InputGradientPenalty, SAMOptimizer
)


def setup_data_loaders(config: ExperimentConfig) -> tuple:
    """
    Set up training and test data loaders based on config.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Get transforms
    if config.use_augmentation:
        train_transform = get_data_augmentation_transforms(config.dataset)
    else:
        train_transform = get_standard_transforms(config.dataset)
    
    test_transform = get_standard_transforms(config.dataset)
    
    # Load datasets
    if config.dataset == 'mnist':
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=test_transform
        )
    elif config.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    # Subset if requested
    if config.train_subset_size is not None:
        indices = np.random.RandomState(42).permutation(len(train_dataset))[:config.train_subset_size]
        train_dataset = Subset(train_dataset, indices)
    
    if config.test_subset_size is not None:
        indices = np.random.RandomState(42).permutation(len(test_dataset))[:config.test_subset_size]
        test_dataset = Subset(test_dataset, indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, test_loader


def setup_model_and_optimizer(config: ExperimentConfig, device: torch.device) -> tuple:
    """
    Set up model, optimizer, and loss function based on config.
    
    Args:
        config: Experiment configuration
        device: Device to place model on
    
    Returns:
        Tuple of (model, optimizer, loss_fn, is_sam)
    """
    # Create model
    model = get_vision_model(config.model_type, dropout_rate=config.dropout_rate)
    model = model.to(device)
    
    # Create optimizer
    optimizer_config = {
        'optimizer': config.optimizer,
        'lr': config.lr,
        'weight_decay': config.weight_decay,
        'use_sam': (config.regularization_type == 'sam'),
        'sam_rho': config.sam_rho
    }
    optimizer, is_sam = get_optimizer_with_config(model, optimizer_config)
    
    # Create loss function
    if config.label_smoothing > 0:
        num_classes = 10  # Both MNIST and CIFAR-10 have 10 classes
        loss_fn = LabelSmoothingLoss(smoothing=config.label_smoothing, num_classes=num_classes)
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    return model, optimizer, loss_fn, is_sam


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer) -> dict:
    """
    Load checkpoint if it exists.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
    
    Returns:
        Dictionary with checkpoint data (epoch, metrics, etc.)
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Handle SAM optimizer
    if isinstance(optimizer, SAMOptimizer):
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_checkpoint(checkpoint_path: str, epoch: int, model: nn.Module, optimizer, 
                   metrics: dict, lambda_tracker: LambdaTracker):
    """
    Save training checkpoint.
    
    Args:
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch
        model: Model to save
        optimizer: Optimizer to save
        metrics: Dictionary of metrics to save
        lambda_tracker: Lambda tracker object
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Handle SAM optimizer
    if isinstance(optimizer, SAMOptimizer):
        optimizer_state = optimizer.optimizer.state_dict()
    else:
        optimizer_state = optimizer.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_state,
        'metrics': metrics,
        'lambda_history': lambda_tracker.get_history()
    }
    
    torch.save(checkpoint, checkpoint_path)


def train_single_epoch(model: nn.Module, train_loader: DataLoader, 
                      optimizer, loss_fn, device: torch.device, 
                      is_sam: bool = False, igp_regularizer = None) -> dict:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        is_sam: Whether using SAM optimizer
        igp_regularizer: Input gradient penalty regularizer (if using)
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        if is_sam:
            # SAM requires two forward-backward passes
            # First pass: compute gradient and find adversarial perturbation
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Add IGP if using
            if igp_regularizer is not None:
                loss = loss + igp_regularizer(model, inputs, labels, nn.CrossEntropyLoss(reduction='none'))
            
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # Second pass: compute gradient at perturbed point
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            if igp_regularizer is not None:
                loss = loss + igp_regularizer(model, inputs, labels, nn.CrossEntropyLoss(reduction='none'))
            
            loss.backward()
            optimizer.second_step(zero_grad=True)
            
        else:
            # Standard training
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Add IGP if using
            if igp_regularizer is not None:
                loss = loss + igp_regularizer(model, inputs, labels, nn.CrossEntropyLoss(reduction='none'))
            
            loss.backward()
            optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += inputs.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }


def measure_lambda(model: nn.Module, train_loader: DataLoader, 
                  device: torch.device, K_dirs: int, max_order: int,
                  store_full_data: bool = False) -> dict:
    """
    Measure lambda on a batch from training set.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        device: Device
        K_dirs: Number of random directions
        max_order: Maximum derivative order
        store_full_data: Whether to return full derivative data
    
    Returns:
        Dictionary with lambda statistics
    """
    # Get a batch
    inputs, labels = next(iter(train_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    
    # Estimate lambda
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    lambda_result = estimate_lambda_per_direction(
        model=model,
        inputs=inputs,
        labels=labels,
        loss_fn=loss_fn,
        max_order=max_order,
        K_dirs=K_dirs,
        return_derivatives=store_full_data
    )
    
    return lambda_result


def run_single_experiment(config: ExperimentConfig, output_dir: str, 
                         device: torch.device = None) -> dict:
    """
    Run a single proof-of-concept experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Directory to save results
        device: Device to run on (default: auto-detect)
    
    Returns:
        Dictionary with final metrics and file paths
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"Running Experiment: {config.method_name} on {config.dataset}")
    print(f"{'='*80}\n")
    
    # Create output directory
    exp_dir = Path(output_dir) / config.dataset / config.method_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(str(exp_dir / 'config.json'))
    
    # Set up checkpoint
    checkpoint_dir = Path(output_dir).parent / 'progress' / config.dataset / config.method_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(checkpoint_dir / 'checkpoint.pt')
    
    # Set up data loaders
    train_loader, test_loader = setup_data_loaders(config)
    
    # Set up model and optimizer
    model, optimizer, loss_fn, is_sam = setup_model_and_optimizer(config, device)
    
    # Set up IGP regularizer if needed
    igp_regularizer = None
    if config.regularization_type == 'igp':
        igp_regularizer = InputGradientPenalty(scale=config.igp_scale)
    
    # Set up lambda tracker
    lambda_tracker = LambdaTracker(store_full_data=True)
    
    # Set up measurement schedule
    measurement_epochs = adaptive_measurement_schedule(config.epochs)
    
    # Metrics storage
    all_epochs = []
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    eces = []
    lambda_means_all_epochs = [None] * (config.epochs + 1)  # Include epoch 0
    lambda_stds_all_epochs = [None] * (config.epochs + 1)
    
    # Check for existing checkpoint
    start_epoch = 0
    checkpoint = load_checkpoint(checkpoint_path, model, optimizer)
    if checkpoint is not None:
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        
        # Restore metrics
        if 'metrics' in checkpoint:
            all_epochs = checkpoint['metrics'].get('epochs', [])
            train_accuracies = checkpoint['metrics'].get('train_accuracies', [])
            test_accuracies = checkpoint['metrics'].get('test_accuracies', [])
            train_losses = checkpoint['metrics'].get('train_losses', [])
            test_losses = checkpoint['metrics'].get('test_losses', [])
            eces = checkpoint['metrics'].get('eces', [])
        
        # Restore lambda tracker
        if 'lambda_history' in checkpoint:
            history = checkpoint['lambda_history']
            for i, epoch in enumerate(history['epochs']):
                lambda_tracker.record(
                    epoch=epoch,
                    lambda_mean=history['lambda_means'][i],
                    lambda_std=history['lambda_stds'][i],
                    lambda_values=history.get('lambda_values_per_epoch', [[]])[i],
                    derivatives_per_dir=history.get('derivatives_per_epoch', [[]])[i]
                )
    
    # Initial measurement at epoch 0 if starting from scratch
    if start_epoch == 0 and 0 in measurement_epochs:
        print(f"Measuring lambda at epoch 0...")
        lambda_result = measure_lambda(model, train_loader, device, config.K_dirs, config.max_order, store_full_data=True)
        if lambda_result['lambda_mean'] is not None:
            lambda_tracker.record(
                epoch=0,
                lambda_mean=lambda_result['lambda_mean'],
                lambda_std=lambda_result['lambda_std'],
                lambda_values=lambda_result['lambda_values'],
                derivatives_per_dir=lambda_result.get('derivatives_per_dir', [])
            )
            lambda_means_all_epochs[0] = lambda_result['lambda_mean']
            lambda_stds_all_epochs[0] = lambda_result['lambda_std']
    
    # Training loop
    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_metrics = train_single_epoch(
            model, train_loader, optimizer, loss_fn, device, is_sam, igp_regularizer
        )
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, loss_fn, device, compute_calibration=True)
        
        # Record metrics
        all_epochs.append(epoch + 1)
        train_accuracies.append(train_metrics['accuracy'])
        test_accuracies.append(test_metrics['accuracy'])
        train_losses.append(train_metrics['loss'])
        test_losses.append(test_metrics['loss'])
        eces.append(test_metrics.get('ece', 0.0))
        
        # Measure lambda if scheduled
        if (epoch + 1) in measurement_epochs:
            print(f"  Measuring lambda at epoch {epoch + 1}...")
            lambda_result = measure_lambda(model, train_loader, device, config.K_dirs, config.max_order, store_full_data=True)
            if lambda_result['lambda_mean'] is not None:
                lambda_tracker.record(
                    epoch=epoch + 1,
                    lambda_mean=lambda_result['lambda_mean'],
                    lambda_std=lambda_result['lambda_std'],
                    lambda_values=lambda_result['lambda_values'],
                    derivatives_per_dir=lambda_result.get('derivatives_per_dir', [])
                )
                lambda_means_all_epochs[epoch + 1] = lambda_result['lambda_mean']
                lambda_stds_all_epochs[epoch + 1] = lambda_result['lambda_std']
        
        # Print progress
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{config.epochs} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Test Acc: {test_metrics['accuracy']:.4f} | "
              f"Loss: {test_metrics['loss']:.4f} | "
              f"ECE: {test_metrics.get('ece', 0.0):.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
            metrics_dict = {
                'epochs': all_epochs,
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'eces': eces
            }
            save_checkpoint(checkpoint_path, epoch, model, optimizer, metrics_dict, lambda_tracker)
            print(f"  Checkpoint saved at epoch {epoch + 1}")
    
    # Save final results
    print("\nSaving results...")
    
    # Save lambda data
    lambda_data_path = str(exp_dir / 'lambda_data.npz')
    
    # Get lambda values aligned with measurement epochs
    lambda_history = lambda_tracker.get_history()
    measured_epochs = lambda_history['epochs']
    train_accs_at_measurements = [train_accuracies[e-1] if e > 0 and e <= len(train_accuracies) else 0.0 for e in measured_epochs]
    test_accs_at_measurements = [test_accuracies[e-1] if e > 0 and e <= len(test_accuracies) else 0.0 for e in measured_epochs]
    train_losses_at_measurements = [train_losses[e-1] if e > 0 and e <= len(train_losses) else 0.0 for e in measured_epochs]
    test_losses_at_measurements = [test_losses[e-1] if e > 0 and e <= len(test_losses) else 0.0 for e in measured_epochs]
    
    save_lambda_data(
        tracker=lambda_tracker,
        output_path=lambda_data_path,
        train_accuracies=train_accs_at_measurements,
        test_accuracies=test_accs_at_measurements,
        train_losses=train_losses_at_measurements,
        test_losses=test_losses_at_measurements
    )
    
    # Save training metrics CSV
    metrics_csv_path = str(exp_dir / 'training_metrics.csv')
    save_training_metrics(
        epochs=all_epochs,
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
        train_losses=train_losses,
        test_losses=test_losses,
        output_path=metrics_csv_path,
        eces=eces,
        lambda_means=[lambda_means_all_epochs[e] for e in all_epochs],
        lambda_stds=[lambda_stds_all_epochs[e] for e in all_epochs]
    )
    
    # Save summary
    final_results = {
        'method_name': config.method_name,
        'dataset': config.dataset,
        'final_train_accuracy': train_accuracies[-1] if train_accuracies else 0.0,
        'final_test_accuracy': test_accuracies[-1] if test_accuracies else 0.0,
        'final_generalization_gap': (train_accuracies[-1] - test_accuracies[-1]) if train_accuracies else 0.0,
        'final_ece': eces[-1] if eces else 0.0,
        'final_lambda_mean': lambda_history['lambda_means'][-1] if lambda_history['lambda_means'] else None,
        'final_lambda_std': lambda_history['lambda_stds'][-1] if lambda_history['lambda_stds'] else None
    }
    
    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Clean up checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint cleaned up")
    
    print(f"\nExperiment completed!")
    print(f"Results saved to: {exp_dir}")
    print(f"Final test accuracy: {final_results['final_test_accuracy']:.4f}")
    print(f"Final lambda: {final_results['final_lambda_mean']:.4f} ± {final_results['final_lambda_std']:.4f}")
    
    return {
        'config': config,
        'results': final_results,
        'lambda_data_path': lambda_data_path,
        'metrics_csv_path': metrics_csv_path,
        'output_dir': str(exp_dir)
    }
