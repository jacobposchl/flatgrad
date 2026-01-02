"""
Proof of Concept: Lambda Measurability and Stability in Neural Networks

This experiment validates that λ (curvature rate) is:
1. Measurable using per-direction derivatives
2. Stable across random directions (low variance)
3. Interpretable in vision tasks (MNIST, CIFAR-10)

Two experiments:
1. MNIST training: Track λ evolution across training epochs
2. CIFAR-10 training: Track λ evolution on higher-dimensional inputs (32×32×3 vs 28×28×1)
"""

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from pathlib import Path
import sys
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flatgrad.sampling.vision_models import create_vision_model
from flatgrad.sampling.lambda_estimation import estimate_lambda_per_direction
from flatgrad.sampling.training import train_epoch, evaluate, LambdaTracker, RegTracker
from flatgrad.sampling.regularizers import compute_lambda_regularizer, compute_lambda_target_regularizer
from experiments.proof_of_concept.helpers.visualization import (
    plot_lambda_evolution, 
    plot_derivative_profile,
    plot_cross_dataset_comparison,
    plot_lambda_distribution
)
from experiments.proof_of_concept.helpers.reg_comparison_plots import (
    plot_lambda_evolution_multi_reg,
    plot_all_metrics_vs_reg_scale,
    plot_reg_magnitude_evolution,
    plot_reg_magnitude_vs_scale
)


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
K_DIRS = 5  # Number of directions for lambda estimation
MAX_ORDER = 4  # Maximum derivative order (1-4)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def loss_fn_with_reduction_none(logits, labels, reduction='none'):
    """Wrapper for F.cross_entropy to handle reduction parameter."""
    return F.cross_entropy(logits, labels, reduction=reduction)


def save_results_to_file(mnist_results_by_reg, cifar_results_by_reg, n_epochs):
    """Save comprehensive results to text file for all regularization scales."""
    from pathlib import Path
    output_dir = Path('results/proof_of_concept')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'results.txt'
    
    reg_scales = sorted(mnist_results_by_reg.keys())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PROOF OF CONCEPT: Lambda Measurability in Neural Networks - RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training Epochs: {n_epochs}\n")
        f.write(f"K_dirs (directions): {K_DIRS}\n")
        f.write(f"Max derivative order: {MAX_ORDER}\n")
        f.write(f"Regularization scales tested: {reg_scales}\n\n")
        
        # Summary table across all reg scales
        f.write("="*80 + "\n")
        f.write("SUMMARY ACROSS REGULARIZATION SCALES\n")
        f.write("="*80 + "\n\n")
        
        f.write("MNIST Summary:\n")
        f.write(f"{'Reg Scale':<12} {'Test Acc':<12} {'Gen Gap':<12} {'Test ECE':<12} {'Final λ':<15} {'Mean Reg Ratio':<15}\n")
        f.write("-"*80 + "\n")
        for reg_scale in reg_scales:
            mnist_results = mnist_results_by_reg[reg_scale]
            test_acc = mnist_results['final_test']['accuracy']
            gen_gap = mnist_results['final_train']['accuracy'] - mnist_results['final_test']['accuracy']
            test_ece = mnist_results['final_test'].get('ece', 0.0)
            lambda_hist = mnist_results['tracker'].get_history()
            final_lambda = lambda_hist['lambda_means'][-1] if len(lambda_hist['lambda_means']) > 0 else 0.0
            reg_ratio = 0.0
            if mnist_results.get('reg_tracker') is not None:
                reg_hist = mnist_results['reg_tracker'].get_history()
                if len(reg_hist['reg_ratios']) > 0:
                    reg_ratio = np.mean(reg_hist['reg_ratios'])
            f.write(f"{reg_scale:<12.3f} {test_acc:<12.4f} {gen_gap:<12.4f} {test_ece:<12.6f} {final_lambda:<15.6f} {reg_ratio:<15.6f}\n")
        f.write("\n")
        
        f.write("CIFAR-10 Summary:\n")
        f.write(f"{'Reg Scale':<12} {'Test Acc':<12} {'Gen Gap':<12} {'Test ECE':<12} {'Final λ':<15} {'Mean Reg Ratio':<15}\n")
        f.write("-"*80 + "\n")
        for reg_scale in reg_scales:
            cifar_results = cifar_results_by_reg[reg_scale]
            test_acc = cifar_results['final_test']['accuracy']
            gen_gap = cifar_results['final_train']['accuracy'] - cifar_results['final_test']['accuracy']
            test_ece = cifar_results['final_test'].get('ece', 0.0)
            lambda_hist = cifar_results['tracker'].get_history()
            final_lambda = lambda_hist['lambda_means'][-1] if len(lambda_hist['lambda_means']) > 0 else 0.0
            reg_ratio = 0.0
            if cifar_results.get('reg_tracker') is not None:
                reg_hist = cifar_results['reg_tracker'].get_history()
                if len(reg_hist['reg_ratios']) > 0:
                    reg_ratio = np.mean(reg_hist['reg_ratios'])
            f.write(f"{reg_scale:<12.3f} {test_acc:<12.4f} {gen_gap:<12.4f} {test_ece:<12.6f} {final_lambda:<15.6f} {reg_ratio:<15.6f}\n")
        f.write("\n\n")
        
        # Detailed results for each reg scale
        for reg_scale in reg_scales:
            mnist_results = mnist_results_by_reg[reg_scale]
            cifar_results = cifar_results_by_reg[reg_scale]
            
            f.write("="*80 + "\n")
            f.write(f"REGULARIZATION SCALE: {reg_scale}\n")
            f.write("="*80 + "\n\n")
            
            # MNIST Results
            f.write("--- MNIST RESULTS ---\n\n")
            
            # Lambda statistics
            f.write("Lambda (λ) Statistics:\n")
            lambda_hist = mnist_results['tracker'].get_history()
            if len(lambda_hist['epochs']) > 0:
                initial_lambda = lambda_hist['lambda_means'][0]
                final_lambda = lambda_hist['lambda_means'][-1]
                lambda_change = ((final_lambda - initial_lambda) / abs(initial_lambda)) * 100 if initial_lambda != 0 else 0
                
                f.write(f"  Initial λ (epoch 0):  {initial_lambda:.6f} ± {lambda_hist['lambda_stds'][0]:.6f}\n")
                f.write(f"  Final λ (epoch {n_epochs}):   {final_lambda:.6f} ± {lambda_hist['lambda_stds'][-1]:.6f}\n")
                f.write(f"  Change:              {lambda_change:+.2f}%\n")
                f.write(f"  Mean across training: {np.mean(lambda_hist['lambda_means']):.6f}\n")
                f.write(f"  Std across training:  {np.std(lambda_hist['lambda_means']):.6f}\n")
            else:
                f.write("  No valid lambda measurements\n")
            f.write("\n")
            
            # Regularization statistics
            if mnist_results.get('reg_tracker') is not None:
                f.write("Regularization Statistics:\n")
                reg_hist = mnist_results['reg_tracker'].get_history()
                if len(reg_hist['epochs']) > 0:
                    f.write(f"  Mean Reg Loss: {np.mean(reg_hist['reg_losses']):.6f}\n")
                    f.write(f"  Mean Main Loss: {np.mean(reg_hist['main_losses']):.6f}\n")
                    f.write(f"  Mean Reg Ratio: {np.mean(reg_hist['reg_ratios']):.6f}\n")
                    f.write(f"  Max Reg Ratio: {np.max(reg_hist['reg_ratios']):.6f}\n")
                f.write("\n")
            
            # Accuracy statistics
            f.write("Accuracy Statistics:\n")
            if len(mnist_results['train_history']['accuracy']) > 0:
                f.write(f"  Initial Train Acc:  {mnist_results['train_history']['accuracy'][0]:.4f}\n")
                f.write(f"  Final Train Acc:    {mnist_results['final_train']['accuracy']:.4f}\n")
                f.write(f"  Final Test Acc:     {mnist_results['final_test']['accuracy']:.4f}\n")
                f.write(f"  Generalization Gap: {(mnist_results['final_train']['accuracy'] - mnist_results['final_test']['accuracy']):.4f}\n")
            f.write("\n")
            
            # ECE
            f.write("Calibration (ECE):\n")
            if 'ece' in mnist_results['final_test']:
                f.write(f"  Test ECE:  {mnist_results['final_test']['ece']:.6f}\n")
                f.write(f"  Train ECE: {mnist_results['final_train']['ece']:.6f}\n")
            f.write("\n")
            
            # CIFAR-10 Results
            f.write("--- CIFAR-10 RESULTS ---\n\n")
            
            # Lambda statistics
            f.write("Lambda (λ) Statistics:\n")
            lambda_hist = cifar_results['tracker'].get_history()
            if len(lambda_hist['epochs']) > 0:
                initial_lambda = lambda_hist['lambda_means'][0]
                final_lambda = lambda_hist['lambda_means'][-1]
                lambda_change = ((final_lambda - initial_lambda) / abs(initial_lambda)) * 100 if initial_lambda != 0 else 0
                
                f.write(f"  Initial λ (epoch 0):  {initial_lambda:.6f} ± {lambda_hist['lambda_stds'][0]:.6f}\n")
                f.write(f"  Final λ (epoch {n_epochs}):   {final_lambda:.6f} ± {lambda_hist['lambda_stds'][-1]:.6f}\n")
                f.write(f"  Change:              {lambda_change:+.2f}%\n")
                f.write(f"  Mean across training: {np.mean(lambda_hist['lambda_means']):.6f}\n")
                f.write(f"  Std across training:  {np.std(lambda_hist['lambda_means']):.6f}\n")
            else:
                f.write("  No valid lambda measurements\n")
            f.write("\n")
            
            # Regularization statistics
            if cifar_results.get('reg_tracker') is not None:
                f.write("Regularization Statistics:\n")
                reg_hist = cifar_results['reg_tracker'].get_history()
                if len(reg_hist['epochs']) > 0:
                    f.write(f"  Mean Reg Loss: {np.mean(reg_hist['reg_losses']):.6f}\n")
                    f.write(f"  Mean Main Loss: {np.mean(reg_hist['main_losses']):.6f}\n")
                    f.write(f"  Mean Reg Ratio: {np.mean(reg_hist['reg_ratios']):.6f}\n")
                    f.write(f"  Max Reg Ratio: {np.max(reg_hist['reg_ratios']):.6f}\n")
                f.write("\n")
            
            # Accuracy statistics
            f.write("Accuracy Statistics:\n")
            if len(cifar_results['train_history']['accuracy']) > 0:
                f.write(f"  Initial Train Acc:  {cifar_results['train_history']['accuracy'][0]:.4f}\n")
                f.write(f"  Final Train Acc:    {cifar_results['final_train']['accuracy']:.4f}\n")
                f.write(f"  Final Test Acc:     {cifar_results['final_test']['accuracy']:.4f}\n")
                f.write(f"  Generalization Gap: {(cifar_results['final_train']['accuracy'] - cifar_results['final_test']['accuracy']):.4f}\n")
            f.write("\n")
            
            # ECE
            f.write("Calibration (ECE):\n")
            if 'ece' in cifar_results['final_test']:
                f.write(f"  Test ECE:  {cifar_results['final_test']['ece']:.6f}\n")
                f.write(f"  Train ECE: {cifar_results['final_train']['ece']:.6f}\n")
            f.write("\n\n")
        
        f.write("="*80 + "\n")
    
    # Results saved (suppress print for cleaner output)


def experiment_2_mnist_training(n_epochs=50, target_lambda=None, reg_scale=None):
    """
    Experiment: Measure λ during MNIST training.
    
    Trains a simple ConvNet and measures λ at different epochs to see
    if it stabilizes or changes systematically during training.
    
    Args:
        n_epochs: Number of training epochs (default: 50)
        target_lambda: Target lambda value to regularize towards (None disables)
        reg_scale: Lambda regularization scale (deprecated, use target_lambda instead)
    """
    # Experiment header (minimal output)
    
    set_seed(SEED)
    
    # Load MNIST data (subset for computational efficiency)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    # Use subset for faster experiments
    train_subset = Subset(train_dataset, range(5000))
    test_subset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    # Get a fixed batch for lambda estimation
    lambda_inputs, lambda_labels = next(iter(test_loader))
    lambda_inputs = lambda_inputs.to(DEVICE)
    lambda_labels = lambda_labels.to(DEVICE)
    
    # Create model
    model = create_vision_model('mnist', dropout_rate=0.5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create regularizer function
    regularizer_fn = None
    if target_lambda is not None:
        # Use target lambda regularizer
        # Scale needs to be larger to effectively push lambda toward target
        # The gradient signal is weak because lambda depends on nested derivatives w.r.t. inputs
        # With scale=0.01, lambda wasn't moving toward targets (e.g., target -7.0, final -2.96)
        # Increasing to 0.1 to strengthen the gradient signal while keeping Reg reasonable
        def regularizer_fn(model, inputs, labels, loss_fn):
            return compute_lambda_target_regularizer(
                model, inputs, labels, loss_fn,
                target_lambda=target_lambda,
                max_order=MAX_ORDER, K_dirs=K_DIRS, scale=0.1
            )
    elif reg_scale is not None and reg_scale > 0:
        # Legacy: use old regularizer for backward compatibility
        def regularizer_fn(model, inputs, labels, loss_fn):
            return compute_lambda_regularizer(
                model, inputs, labels, loss_fn,
                start_n=1, end_n=MAX_ORDER, K_dirs=K_DIRS, scale=reg_scale
            )
    
    # Track lambda
    tracker = LambdaTracker()
    
    # Track regularization
    reg_tracker = RegTracker()
    
    # Track training metrics
    train_history = {'accuracy': [], 'loss': []}
    test_history = {'accuracy': [], 'loss': []}
    
    # Measure initial lambda
    result = estimate_lambda_per_direction(
        model, lambda_inputs, lambda_labels,
        loss_fn_with_reduction_none,
        max_order=MAX_ORDER,
        K_dirs=K_DIRS
    )
    if result['lambda_mean'] is not None:
        tracker.record(0, result['lambda_mean'], result['lambda_std'])
        # Save initial lambda distribution
        if len(result['lambda_values']) > 0:
            reg_label = f"target_{target_lambda:.2f}" if target_lambda is not None else f"reg_{reg_scale}" if reg_scale else "no_reg"
            plot_lambda_distribution(
                result['lambda_values'],
                title="MNIST: Lambda Distribution Across Directions (Epoch 0)",
                save_path=f"results/proof_of_concept/mnist/direction_distribution/{reg_label}_epoch0.png",
                reg_scale=target_lambda if target_lambda is not None else reg_scale
            )
    
    # Training loop with progress bar
    reg_label = f"λ_tgt={target_lambda:.2f}" if target_lambda is not None else f"reg={reg_scale}" if reg_scale else "no_reg"
    epoch_pbar = tqdm(range(1, n_epochs + 1), desc=f"MNIST ({reg_label})", 
                      position=1, leave=False,
                      dynamic_ncols=True,
                      mininterval=0.1,
                      file=sys.stdout,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    try:
        for epoch in epoch_pbar:
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer,
                F.cross_entropy, DEVICE, regularizer_fn
            )
            
            # Evaluate
            test_metrics = evaluate(model, test_loader, F.cross_entropy, DEVICE)
            
            # Store metrics
            train_history['accuracy'].append(train_metrics['accuracy'])
            train_history['loss'].append(train_metrics['loss'])
            test_history['accuracy'].append(test_metrics['accuracy'])
            test_history['loss'].append(test_metrics['loss'])
            
            # Track regularization
            has_reg = target_lambda is not None or (reg_scale is not None and reg_scale > 0)
            if has_reg:
                reg_tracker.record(epoch, train_metrics['reg_loss'], train_metrics['main_loss'])
            
            # Measure lambda
            result = estimate_lambda_per_direction(
                model, lambda_inputs, lambda_labels,
                loss_fn_with_reduction_none,
                max_order=MAX_ORDER,
                K_dirs=K_DIRS
            )
            
            if result['lambda_mean'] is not None:
                tracker.record(epoch, result['lambda_mean'], result['lambda_std'])
            
            # Update progress bar (single update per epoch)
            reg_ratio = train_metrics['reg_loss'] / train_metrics['main_loss'] if has_reg and train_metrics['main_loss'] > 0 else 0.0
            
            postfix = {
                'Train': f"{train_metrics['accuracy']:.3f}",
                'Test': f"{test_metrics['accuracy']:.3f}",
                'λ': f"{result['lambda_mean']:.3f}" if result['lambda_mean'] is not None else "N/A",
            }
            if target_lambda is not None:
                postfix['λ_tgt'] = f"{target_lambda:.2f}"
            if has_reg:
                postfix['Reg'] = f"{reg_ratio:.3f}"
            
            epoch_pbar.set_postfix(postfix, refresh=True)
    finally:
        epoch_pbar.close()
    
    # Summary (suppressed for cleaner output, available in results.txt)
    # tracker.print_summary()
    # if reg_scale > 0:
    #     reg_tracker.print_summary()
    
    epoch_pbar.close()  # Close progress bar before final operations
    
    # Compute final metrics with ECE
    final_test_metrics = evaluate(model, test_loader, F.cross_entropy, DEVICE, compute_calibration=True)
    final_train_metrics = evaluate(model, train_loader, F.cross_entropy, DEVICE, compute_calibration=True)
    
    # Save final lambda distribution
    final_result = estimate_lambda_per_direction(
        model, lambda_inputs, lambda_labels,
        loss_fn_with_reduction_none,
        max_order=MAX_ORDER,
        K_dirs=K_DIRS
    )
    if len(final_result['lambda_values']) > 0:
        reg_label = f"target_{target_lambda:.2f}" if target_lambda is not None else f"reg_{reg_scale}" if reg_scale else "no_reg"
        plot_lambda_distribution(
            final_result['lambda_values'],
            title=f"MNIST: Lambda Distribution Across Directions (Epoch {n_epochs})",
            save_path=f"results/proof_of_concept/mnist/direction_distribution/{reg_label}_final.png",
            reg_scale=target_lambda if target_lambda is not None else reg_scale
        )
    
    # Visualize
    plot_lambda_evolution(
        tracker.get_history(),
        title="MNIST: Lambda Evolution During Training",
        save_path=f"results/proof_of_concept/mnist/lambda_evolution/reg_{reg_scale}.png"
    )
    
    return {
        'tracker': tracker,
        'reg_tracker': reg_tracker if (target_lambda is not None or (reg_scale is not None and reg_scale > 0)) else None,
        'train_history': train_history,
        'test_history': test_history,
        'final_train': final_train_metrics,
        'final_test': final_test_metrics,
        'reg_scale': reg_scale
    }


def experiment_3_cifar10_training(n_epochs=50, target_lambda=None, reg_scale=None):
    """
    Experiment 2: Track lambda evolution during CIFAR-10 training.
    
    CIFAR-10 has 3072-dimensional inputs (32×32×3) vs MNIST's 784 dimensions,
    testing whether lambda remains measurable in more complex scenarios.
    
    Args:
        n_epochs: Number of training epochs (default: 50)
        target_lambda: Target lambda value to regularize towards (None disables)
        reg_scale: Lambda regularization scale (deprecated, use target_lambda instead)
    """
    # Experiment header (minimal output)
    
    set_seed(SEED)
    
    # Load CIFAR-10 data (subset for computational efficiency)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        './data', train=False, download=True, transform=transform
    )
    
    # Use subset for faster experiments
    train_subset = Subset(train_dataset, range(5000))
    test_subset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    # Get a fixed batch for lambda estimation
    lambda_inputs, lambda_labels = next(iter(test_loader))
    lambda_inputs = lambda_inputs.to(DEVICE)
    lambda_labels = lambda_labels.to(DEVICE)
    
    # Create model
    model = create_vision_model('cifar10', dropout_rate=0.5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create regularizer function
    regularizer_fn = None
    if target_lambda is not None:
        # Use target lambda regularizer
        # Scale needs to be larger to effectively push lambda toward target
        # The gradient signal is weak because lambda depends on nested derivatives w.r.t. inputs
        # With scale=0.01, lambda wasn't moving toward targets (e.g., target -7.0, final -2.96)
        # Increasing to 0.1 to strengthen the gradient signal while keeping Reg reasonable
        def regularizer_fn(model, inputs, labels, loss_fn):
            return compute_lambda_target_regularizer(
                model, inputs, labels, loss_fn,
                target_lambda=target_lambda,
                max_order=MAX_ORDER, K_dirs=K_DIRS, scale=0.1
            )
    elif reg_scale is not None and reg_scale > 0:
        # Legacy: use old regularizer for backward compatibility
        def regularizer_fn(model, inputs, labels, loss_fn):
            return compute_lambda_regularizer(
                model, inputs, labels, loss_fn,
                start_n=1, end_n=MAX_ORDER, K_dirs=K_DIRS, scale=reg_scale
            )
    
    # Track lambda
    tracker = LambdaTracker()
    
    # Track regularization
    reg_tracker = RegTracker()
    
    # Track training metrics
    train_history = {'accuracy': [], 'loss': []}
    test_history = {'accuracy': [], 'loss': []}
    
    # Measure initial lambda
    result = estimate_lambda_per_direction(
        model, lambda_inputs, lambda_labels,
        loss_fn_with_reduction_none,
        max_order=MAX_ORDER,
        K_dirs=K_DIRS
    )
    if result['lambda_mean'] is not None:
        tracker.record(0, result['lambda_mean'], result['lambda_std'])
        # Save initial lambda distribution
        if len(result['lambda_values']) > 0:
            reg_label = f"target_{target_lambda:.2f}" if target_lambda is not None else f"reg_{reg_scale}" if reg_scale else "no_reg"
            plot_lambda_distribution(
                result['lambda_values'],
                title="CIFAR-10: Lambda Distribution Across Directions (Epoch 0)",
                save_path=f"results/proof_of_concept/cifar10/direction_distribution/{reg_label}_epoch0.png",
                reg_scale=target_lambda if target_lambda is not None else reg_scale
            )
    
    # Training loop with progress bar
    reg_label = f"λ_tgt={target_lambda:.2f}" if target_lambda is not None else f"reg={reg_scale}" if reg_scale else "no_reg"
    epoch_pbar = tqdm(range(1, n_epochs + 1), desc=f"CIFAR-10 ({reg_label})", 
                      position=1, leave=False,
                      dynamic_ncols=True,
                      mininterval=0.1,
                      file=sys.stdout,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    try:
        for epoch in epoch_pbar:
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer,
                F.cross_entropy, DEVICE, regularizer_fn
            )
            
            # Evaluate
            test_metrics = evaluate(model, test_loader, F.cross_entropy, DEVICE)
            
            # Store metrics
            train_history['accuracy'].append(train_metrics['accuracy'])
            train_history['loss'].append(train_metrics['loss'])
            test_history['accuracy'].append(test_metrics['accuracy'])
            test_history['loss'].append(test_metrics['loss'])
            
            # Track regularization
            has_reg = target_lambda is not None or (reg_scale is not None and reg_scale > 0)
            if has_reg:
                reg_tracker.record(epoch, train_metrics['reg_loss'], train_metrics['main_loss'])
            
            # Measure lambda
            result = estimate_lambda_per_direction(
                model, lambda_inputs, lambda_labels,
                loss_fn_with_reduction_none,
                max_order=MAX_ORDER,
                K_dirs=K_DIRS
            )
            
            if result['lambda_mean'] is not None:
                tracker.record(epoch, result['lambda_mean'], result['lambda_std'])
            
            # Update progress bar (single update per epoch)
            reg_ratio = train_metrics['reg_loss'] / train_metrics['main_loss'] if has_reg and train_metrics['main_loss'] > 0 else 0.0
            
            postfix = {
                'Train': f"{train_metrics['accuracy']:.3f}",
                'Test': f"{test_metrics['accuracy']:.3f}",
                'λ': f"{result['lambda_mean']:.3f}" if result['lambda_mean'] is not None else "N/A",
            }
            if target_lambda is not None:
                postfix['λ_tgt'] = f"{target_lambda:.2f}"
            if has_reg:
                postfix['Reg'] = f"{reg_ratio:.3f}"
            
            epoch_pbar.set_postfix(postfix, refresh=True)
    finally:
        epoch_pbar.close()
    
    # Summary (suppressed for cleaner output, available in results.txt)
    # tracker.print_summary()
    # if reg_scale > 0:
    #     reg_tracker.print_summary()
    
    epoch_pbar.close()  # Close progress bar before final operations
    
    # Compute final metrics with ECE
    final_test_metrics = evaluate(model, test_loader, F.cross_entropy, DEVICE, compute_calibration=True)
    final_train_metrics = evaluate(model, train_loader, F.cross_entropy, DEVICE, compute_calibration=True)
    
    # Save final lambda distribution
    final_result = estimate_lambda_per_direction(
        model, lambda_inputs, lambda_labels,
        loss_fn_with_reduction_none,
        max_order=MAX_ORDER,
        K_dirs=K_DIRS
    )
    if len(final_result['lambda_values']) > 0:
        reg_label = f"target_{target_lambda:.2f}" if target_lambda is not None else f"reg_{reg_scale}" if reg_scale else "no_reg"
        plot_lambda_distribution(
            final_result['lambda_values'],
            title=f"CIFAR-10: Lambda Distribution Across Directions (Epoch {n_epochs})",
            save_path=f"results/proof_of_concept/cifar10/direction_distribution/{reg_label}_final.png",
            reg_scale=target_lambda if target_lambda is not None else reg_scale
        )
    
    # Visualize
    plot_lambda_evolution(
        tracker.get_history(),
        title="CIFAR-10: Lambda Evolution During Training",
        save_path=f"results/proof_of_concept/cifar10/lambda_evolution/reg_{reg_scale}.png"
    )
    
    return {
        'tracker': tracker,
        'reg_tracker': reg_tracker if (target_lambda is not None or (reg_scale is not None and reg_scale > 0)) else None,
        'train_history': train_history,
        'test_history': test_history,
        'final_train': final_train_metrics,
        'final_test': final_test_metrics,
        'reg_scale': reg_scale
    }


def main():
    """Run all proof of concept experiments."""
    parser = argparse.ArgumentParser(description='Proof of Concept: Lambda Measurability')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--target-lambda', type=float, nargs='+', default=None,
                       help='Target lambda values to test (e.g., --target-lambda -2.0 -1.5 -1.0 -0.5 0.0)')
    parser.add_argument('--reg-scale', type=float, nargs='+', default=None,
                       help='Legacy: Lambda regularization scales to test (deprecated, use --target-lambda)')
    args = parser.parse_args()
    
    # Determine which mode to use
    if args.target_lambda is not None:
        # New mode: target lambda grid search
        target_lambdas = args.target_lambda if isinstance(args.target_lambda, list) else [args.target_lambda]
        reg_scales = None
        print("\n" + "="*80)
        print("PROOF OF CONCEPT: Lambda Target Regularization")
        print("="*80)
        print(f"Device: {DEVICE} | Epochs: {args.epochs} | Target Lambdas: {target_lambdas}")
        print("="*80 + "\n")
    elif args.reg_scale is not None:
        # Legacy mode: reg scale grid search
        reg_scales = args.reg_scale if isinstance(args.reg_scale, list) else [args.reg_scale]
        target_lambdas = None
        print("\n" + "="*80)
        print("PROOF OF CONCEPT: Lambda Measurability in Neural Networks")
        print("="*80)
        print(f"Device: {DEVICE} | Epochs: {args.epochs} | Reg Scales: {reg_scales}")
        print("="*80 + "\n")
    else:
        # Default: use target lambda grid
        
        target_lambdas = [-4.0, -2.0, -1.0, 0, 1.0, 2.0, 4.0]
        reg_scales = None
        print("\n" + "="*80)
        print("PROOF OF CONCEPT: Lambda Target Regularization")
        print("="*80)
        print(f"Device: {DEVICE} | Epochs: {args.epochs} | Target Lambdas: {target_lambdas} (default)")
        print("="*80 + "\n")
    
    # Run experiments
    mnist_results_by_reg = {}
    cifar_results_by_reg = {}
    
    if target_lambdas is not None:
        # Target lambda mode
        reg_pbar = tqdm(target_lambdas, desc="Target Lambdas", position=0, leave=True, 
                        dynamic_ncols=True,
                        mininterval=0.5,
                        file=sys.stdout,
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for target_lambda in reg_pbar:
            reg_pbar.set_description(f"Target λ: {target_lambda:.2f}", refresh=False)
            mnist_results = experiment_2_mnist_training(n_epochs=args.epochs, target_lambda=target_lambda)
            cifar_results = experiment_3_cifar10_training(n_epochs=args.epochs, target_lambda=target_lambda)
            
            mnist_results_by_reg[target_lambda] = mnist_results
            cifar_results_by_reg[target_lambda] = cifar_results
    else:
        # Legacy reg scale mode
        reg_pbar = tqdm(reg_scales, desc="Reg Scales", position=0, leave=True, 
                        dynamic_ncols=True,
                        mininterval=0.5,
                        file=sys.stdout,
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for reg_scale in reg_pbar:
            reg_pbar.set_description(f"Reg Scale: {reg_scale}", refresh=False)
            mnist_results = experiment_2_mnist_training(n_epochs=args.epochs, reg_scale=reg_scale)
            cifar_results = experiment_3_cifar10_training(n_epochs=args.epochs, reg_scale=reg_scale)
            
            mnist_results_by_reg[reg_scale] = mnist_results
            cifar_results_by_reg[reg_scale] = cifar_results
    
    # Save detailed results for all reg scales
    print("\n" + "="*80)
    print("Saving results and generating plots...")
    print("="*80)
    save_results_to_file(mnist_results_by_reg, cifar_results_by_reg, args.epochs)
    
    # Create comparison plots
    plot_pbar = tqdm(total=8, desc="Generating plots", position=0, leave=True)
    
    # Lambda evolution plots with multiple reg scales
    plot_lambda_evolution_multi_reg(
        mnist_results_by_reg,
        dataset_name="MNIST",
        save_path="results/proof_of_concept/mnist/lambda_evolution/multi_reg_comparison.png"
    )
    plot_pbar.update(1)
    
    plot_lambda_evolution_multi_reg(
        cifar_results_by_reg,
        dataset_name="CIFAR10",
        save_path="results/proof_of_concept/cifar10/lambda_evolution/multi_reg_comparison.png"
    )
    plot_pbar.update(1)
    
    # Metric vs reg scale plots
    plot_all_metrics_vs_reg_scale(
        mnist_results_by_reg,
        dataset_name="MNIST"
    )
    plot_pbar.update(3)  # 3 plots per dataset
    
    plot_all_metrics_vs_reg_scale(
        cifar_results_by_reg,
        dataset_name="CIFAR10"
    )
    plot_pbar.update(3)  # 3 plots per dataset
    
    # Regularization magnitude plots
    plot_reg_magnitude_evolution(
        mnist_results_by_reg,
        dataset_name="MNIST",
        save_path="results/proof_of_concept/mnist/reg_magnitude_evolution.png"
    )
    plot_pbar.update(1)
    
    plot_reg_magnitude_evolution(
        cifar_results_by_reg,
        dataset_name="CIFAR10",
        save_path="results/proof_of_concept/cifar10/reg_magnitude_evolution.png"
    )
    plot_pbar.update(1)
    
    plot_reg_magnitude_vs_scale(
        mnist_results_by_reg,
        dataset_name="MNIST",
        save_path="results/proof_of_concept/mnist/reg_magnitude_vs_scale.png"
    )
    plot_pbar.update(1)
    
    plot_reg_magnitude_vs_scale(
        cifar_results_by_reg,
        dataset_name="CIFAR10",
        save_path="results/proof_of_concept/cifar10/reg_magnitude_vs_scale.png"
    )
    plot_pbar.update(1)
    plot_pbar.close()
    
    print("\n" + "="*80)
    print("✓ ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Results saved to: results/proof_of_concept/results.txt")
    print(f"Plots saved to: results/proof_of_concept/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
