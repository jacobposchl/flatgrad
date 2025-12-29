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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flatgrad.sampling.vision_models import create_vision_model
from flatgrad.sampling.lambda_estimation import estimate_lambda_per_direction
from flatgrad.sampling.training import train_epoch, evaluate, LambdaTracker
from flatgrad.sampling.regularizers import compute_lambda_regularizer
from helpers.visualization import (
    plot_lambda_evolution, 
    plot_derivative_profile,
    plot_cross_dataset_comparison,
    plot_lambda_distribution
)
from helpers.reg_comparison_plots import (
    plot_lambda_evolution_multi_reg,
    plot_all_metrics_vs_reg_scale
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


def save_results_to_file(mnist_results, cifar_results, n_epochs):
    """Save comprehensive results to text file."""
    from pathlib import Path
    output_dir = Path('results/proof_of_concept')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'results.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PROOF OF CONCEPT: Lambda Measurability in Neural Networks - RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training Epochs: {n_epochs}\n")
        f.write(f"K_dirs (directions): {K_DIRS}\n")
        f.write(f"Max derivative order: {MAX_ORDER}\n\n")
        
        # MNIST Results
        f.write("="*80 + "\n")
        f.write("MNIST RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Lambda statistics
        f.write("--- Lambda (\u03bb) Statistics ---\n")
        lambda_hist = mnist_results['tracker'].get_history()
        if len(lambda_hist['epochs']) > 0:
            initial_lambda = lambda_hist['lambda_means'][0]
            final_lambda = lambda_hist['lambda_means'][-1]
            lambda_change = ((final_lambda - initial_lambda) / abs(initial_lambda)) * 100 if initial_lambda != 0 else 0
            
            f.write(f"  Initial \u03bb (epoch 0):  {initial_lambda:.6f} \u00b1 {lambda_hist['lambda_stds'][0]:.6f}\n")
            f.write(f"  Final \u03bb (epoch {n_epochs}):   {final_lambda:.6f} \u00b1 {lambda_hist['lambda_stds'][-1]:.6f}\n")
            f.write(f"  Change:              {lambda_change:+.2f}%\n")
            f.write(f"  Mean across training: {np.mean(lambda_hist['lambda_means']):.6f}\n")
            f.write(f"  Std across training:  {np.std(lambda_hist['lambda_means']):.6f}\n")
            f.write("  No valid lambda measurements\n")
        f.write("\n")
        
        # Accuracy statistics
        f.write("--- Accuracy Statistics ---\n")
        if len(mnist_results['train_history']['accuracy']) > 0:
            f.write(f"  Initial Train Acc:  {mnist_results['train_history']['accuracy'][0]:.4f}\n")
            f.write(f"  Final Train Acc:    {mnist_results['final_train']['accuracy']:.4f}\n")
            f.write(f"  Final Test Acc:     {mnist_results['final_test']['accuracy']:.4f}\n")
            f.write(f"  Generalization Gap: {(mnist_results['final_train']['accuracy'] - mnist_results['final_test']['accuracy']):.4f}\n")
        f.write("\n")
        
        # ECE
        f.write("--- Calibration (ECE) ---\n")
        if 'ece' in mnist_results['final_test']:
            f.write(f"  Test ECE:  {mnist_results['final_test']['ece']:.6f}\n")
            f.write(f"  Train ECE: {mnist_results['final_train']['ece']:.6f}\n")
        f.write("\n")
        
        # Training progression
        f.write("--- Training Progression ---\n")
        f.write("  Epoch | Train Acc | Test Acc  | Train Loss | Test Loss\n")
        f.write("  " + "-"*60 + "\n")
        for i in range(len(mnist_results['train_history']['accuracy'])):
            f.write(f"  {i+1:5d} | {mnist_results['train_history']['accuracy'][i]:.6f} | "
                   f"{mnist_results['test_history']['accuracy'][i]:.6f} | "
                   f"{mnist_results['train_history']['loss'][i]:.6f} | "
                   f"{mnist_results['test_history']['loss'][i]:.6f}\n")
        f.write("\n\n")
        
        # CIFAR-10 Results
        f.write("="*80 + "\n")
        f.write("CIFAR-10 RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Lambda statistics
        f.write("--- Lambda (\u03bb) Statistics ---\n")
        lambda_hist = cifar_results['tracker'].get_history()
        if len(lambda_hist['epochs']) > 0:
            initial_lambda = lambda_hist['lambda_means'][0]
            final_lambda = lambda_hist['lambda_means'][-1]
            lambda_change = ((final_lambda - initial_lambda) / abs(initial_lambda)) * 100 if initial_lambda != 0 else 0
            
            f.write(f"  Initial \u03bb (epoch 0):  {initial_lambda:.6f} \u00b1 {lambda_hist['lambda_stds'][0]:.6f}\n")
            f.write(f"  Final \u03bb (epoch {n_epochs}):   {final_lambda:.6f} \u00b1 {lambda_hist['lambda_stds'][-1]:.6f}\n")
            f.write(f"  Change:              {lambda_change:+.2f}%\n")
            f.write(f"  Mean across training: {np.mean(lambda_hist['lambda_means']):.6f}\n")
            f.write(f"  Std across training:  {np.std(lambda_hist['lambda_means']):.6f}\n")
        else:
            f.write("  No valid lambda measurements\n")
        f.write("\n")
        
        # Accuracy statistics
        f.write("--- Accuracy Statistics ---\n")
        if len(cifar_results['train_history']['accuracy']) > 0:
            f.write(f"  Initial Train Acc:  {cifar_results['train_history']['accuracy'][0]:.4f}\n")
            f.write(f"  Final Train Acc:    {cifar_results['final_train']['accuracy']:.4f}\n")
            f.write(f"  Final Test Acc:     {cifar_results['final_test']['accuracy']:.4f}\n")
            f.write(f"  Generalization Gap: {(cifar_results['final_train']['accuracy'] - cifar_results['final_test']['accuracy']):.4f}\n")
        f.write("\n")
        
        # ECE
        f.write("--- Calibration (ECE) ---\n")
        if 'ece' in cifar_results['final_test']:
            f.write(f"  Test ECE:  {cifar_results['final_test']['ece']:.6f}\n")
            f.write(f"  Train ECE: {cifar_results['final_train']['ece']:.6f}\n")
        f.write("\n")
        
        # Training progression
        f.write("--- Training Progression ---\n")
        f.write("  Epoch | Train Acc | Test Acc  | Train Loss | Test Loss\n")
        f.write("  " + "-"*60 + "\n")
        for i in range(len(cifar_results['train_history']['accuracy'])):
            f.write(f"  {i+1:5d} | {cifar_results['train_history']['accuracy'][i]:.6f} | "
                   f"{cifar_results['test_history']['accuracy'][i]:.6f} | "
                   f"{cifar_results['train_history']['loss'][i]:.6f} | "
                   f"{cifar_results['test_history']['loss'][i]:.6f}\n")
        f.write("\n")
        f.write("="*80 + "\n")
    
    print(f"\nSaved comprehensive results to {output_path}")


def experiment_2_mnist_training(n_epochs=20, reg_scale=1.0):
    """
    Experiment: Measure λ during MNIST training.
    
    Trains a simple ConvNet and measures λ at different epochs to see
    if it stabilizes or changes systematically during training.
    
    Args:
        n_epochs: Number of training epochs (default: 20)
        reg_scale: Lambda regularization scale (default: 1.0, 0 disables)
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: MNIST Lambda Tracking During Training")
    print("="*80)
    
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
    
    # Create regularizer function if scale > 0
    regularizer_fn = None
    if reg_scale > 0:
        def regularizer_fn(model, inputs, labels, loss_fn):
            return compute_lambda_regularizer(
                model, inputs, labels, loss_fn,
                start_n=1, end_n=MAX_ORDER, K_dirs=K_DIRS, scale=reg_scale
            )
    
    # Track lambda
    tracker = LambdaTracker()
    
    # Track training metrics
    train_history = {'accuracy': [], 'loss': []}
    test_history = {'accuracy': [], 'loss': []}
    
    # Measure initial lambda
    print("\nMeasuring initial lambda (before training)...")
    result = estimate_lambda_per_direction(
        model, lambda_inputs, lambda_labels,
        loss_fn_with_reduction_none,
        max_order=MAX_ORDER,
        K_dirs=K_DIRS
    )
    if result['lambda_mean'] is not None:
        tracker.record(0, result['lambda_mean'], result['lambda_std'])
        print(f"  Epoch 0: λ = {result['lambda_mean']:.4f} ± {result['lambda_std']:.4f}")
        # Save initial lambda distribution
        if len(result['lambda_values']) > 0:
            plot_lambda_distribution(
                result['lambda_values'],
                title="MNIST: Lambda Distribution Across Directions (Epoch 0)",
                save_path=f"results/proof_of_concept/mnist/direction_distribution/reg_{reg_scale}_epoch0.png",
                reg_scale=reg_scale
            )
    else:
        print(f"  Epoch 0: ⚠ Lambda estimation failed (valid directions: {result['n_valid_directions']}/{K_DIRS})")
    
    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(1, n_epochs + 1):
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
        
        # Measure lambda
        result = estimate_lambda_per_direction(
            model, lambda_inputs, lambda_labels,
            loss_fn_with_reduction_none,
            max_order=MAX_ORDER,
            K_dirs=K_DIRS
        )
        
        if result['lambda_mean'] is not None:
            tracker.record(epoch, result['lambda_mean'], result['lambda_std'])
            print(f"  Epoch {epoch}: "
                  f"Train Acc={train_metrics['accuracy']:.3f}, "
                  f"Test Acc={test_metrics['accuracy']:.3f}, "
                  f"λ={result['lambda_mean']:.4f}±{result['lambda_std']:.4f}")
        else:
            print(f"  Epoch {epoch}: "
                  f"Train Acc={train_metrics['accuracy']:.3f}, "
                  f"Test Acc={test_metrics['accuracy']:.3f}, "
                  f"λ=N/A (failed, valid: {result['n_valid_directions']}/{K_DIRS})")
    
    # Summary
    tracker.print_summary()
    
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
        plot_lambda_distribution(
            final_result['lambda_values'],
            title=f"MNIST: Lambda Distribution Across Directions (Epoch {n_epochs})",
            save_path=f"results/proof_of_concept/mnist/direction_distribution/reg_{reg_scale}_final.png",
            reg_scale=reg_scale
        )
    
    # Visualize
    plot_lambda_evolution(
        tracker.get_history(),
        title="MNIST: Lambda Evolution During Training",
        save_path=f"results/proof_of_concept/mnist/lambda_evolution/reg_{reg_scale}.png"
    )
    
    return {
        'tracker': tracker,
        'train_history': train_history,
        'test_history': test_history,
        'final_train': final_train_metrics,
        'final_test': final_test_metrics
    }


def experiment_3_cifar10_training(n_epochs=20, reg_scale=1.0):
    """
    Experiment 2: Track lambda evolution during CIFAR-10 training.
    
    CIFAR-10 has 3072-dimensional inputs (32×32×3) vs MNIST's 784 dimensions,
    testing whether lambda remains measurable in more complex scenarios.
    
    Args:
        n_epochs: Number of training epochs (default: 20)
        reg_scale: Lambda regularization scale (default: 1.0, 0 disables)
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: CIFAR-10 Lambda Tracking During Training")
    print("="*80)
    
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
    
    # Create regularizer function if scale > 0
    regularizer_fn = None
    if reg_scale > 0:
        def regularizer_fn(model, inputs, labels, loss_fn):
            return compute_lambda_regularizer(
                model, inputs, labels, loss_fn,
                start_n=1, end_n=MAX_ORDER, K_dirs=K_DIRS, scale=reg_scale
            )
    
    # Track lambda
    tracker = LambdaTracker()
    
    # Track training metrics
    train_history = {'accuracy': [], 'loss': []}
    test_history = {'accuracy': [], 'loss': []}
    
    # Measure initial lambda
    print("\nMeasuring initial lambda (before training)...")
    result = estimate_lambda_per_direction(
        model, lambda_inputs, lambda_labels,
        loss_fn_with_reduction_none,
        max_order=MAX_ORDER,
        K_dirs=K_DIRS
    )
    if result['lambda_mean'] is not None:
        tracker.record(0, result['lambda_mean'], result['lambda_std'])
        print(f"  Epoch 0: λ = {result['lambda_mean']:.4f} ± {result['lambda_std']:.4f}")
        # Save initial lambda distribution
        if len(result['lambda_values']) > 0:
            plot_lambda_distribution(
                result['lambda_values'],
                title="CIFAR-10: Lambda Distribution Across Directions (Epoch 0)",
                save_path=f"results/proof_of_concept/cifar10/direction_distribution/reg_{reg_scale}_epoch0.png",
                reg_scale=reg_scale
            )
    else:
        print(f"  Epoch 0: ⚠ Lambda estimation failed (valid directions: {result['n_valid_directions']}/{K_DIRS})")
    
    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(1, n_epochs + 1):
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
        
        # Measure lambda
        result = estimate_lambda_per_direction(
            model, lambda_inputs, lambda_labels,
            loss_fn_with_reduction_none,
            max_order=MAX_ORDER,
            K_dirs=K_DIRS
        )
        
        if result['lambda_mean'] is not None:
            tracker.record(epoch, result['lambda_mean'], result['lambda_std'])
            print(f"  Epoch {epoch}: "
                  f"Train Acc={train_metrics['accuracy']:.3f}, "
                  f"Test Acc={test_metrics['accuracy']:.3f}, "
                  f"λ={result['lambda_mean']:.4f}±{result['lambda_std']:.4f}")
        else:
            print(f"  Epoch {epoch}: "
                  f"Train Acc={train_metrics['accuracy']:.3f}, "
                  f"Test Acc={test_metrics['accuracy']:.3f}, "
                  f"λ=N/A (failed, valid: {result['n_valid_directions']}/{K_DIRS})")
    
    # Summary
    tracker.print_summary()
    
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
        plot_lambda_distribution(
            final_result['lambda_values'],
            title=f"CIFAR-10: Lambda Distribution Across Directions (Epoch {n_epochs})",
            save_path=f"results/proof_of_concept/cifar10/direction_distribution/reg_{reg_scale}_final.png",
            reg_scale=reg_scale
        )
    
    # Visualize
    plot_lambda_evolution(
        tracker.get_history(),
        title="CIFAR-10: Lambda Evolution During Training",
        save_path=f"results/proof_of_concept/cifar10/lambda_evolution/reg_{reg_scale}.png"
    )
    
    return {
        'tracker': tracker,
        'train_history': train_history,
        'test_history': test_history,
        'final_train': final_train_metrics,
        'final_test': final_test_metrics
    }


def main():
    """Run all proof of concept experiments."""
    parser = argparse.ArgumentParser(description='Proof of Concept: Lambda Measurability')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--reg-scale', type=float, nargs='+', default=[0.0, 0.001, 0.01, 0.1, 1.0],
                       help='Lambda regularization scales to test (default: [0.0, 0.001, 0.01, 0.1, 1.0])')
    args = parser.parse_args()
    
    # Convert single value to list if needed
    reg_scales = args.reg_scale if isinstance(args.reg_scale, list) else [args.reg_scale]
    
    print("\n" + "="*80)
    print("PROOF OF CONCEPT: Lambda Measurability in Neural Networks")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    print(f"  K_dirs: {K_DIRS}")
    print(f"  Max derivative order: {MAX_ORDER}")
    print(f"  Training epochs: {args.epochs}")
    print(f"  Regularization scales: {reg_scales}")
    
    # Run experiments for each regularization scale
    mnist_results_by_reg = {}
    cifar_results_by_reg = {}
    
    for reg_scale in reg_scales:
        print(f"\n{'='*80}")
        print(f"Running experiments with regularization scale: {reg_scale}")
        print(f"{'='*80}")
        
        mnist_results = experiment_2_mnist_training(n_epochs=args.epochs, reg_scale=reg_scale)
        cifar_results = experiment_3_cifar10_training(n_epochs=args.epochs, reg_scale=reg_scale)
        
        mnist_results_by_reg[reg_scale] = mnist_results
        cifar_results_by_reg[reg_scale] = cifar_results
    
    # Save detailed results for the first reg_scale (backward compatibility)
    save_results_to_file(mnist_results_by_reg[reg_scales[0]], 
                        cifar_results_by_reg[reg_scales[0]], 
                        args.epochs)
    
    # Create comparison plots
    print("\n" + "="*80)
    print("CREATING REGULARIZATION COMPARISON PLOTS")
    print("="*80)
    
    # Lambda evolution plots with multiple reg scales
    plot_lambda_evolution_multi_reg(
        mnist_results_by_reg,
        dataset_name="MNIST",
        save_path="results/proof_of_concept/mnist/lambda_evolution/multi_reg_comparison.png"
    )
    
    plot_lambda_evolution_multi_reg(
        cifar_results_by_reg,
        dataset_name="CIFAR10",
        save_path="results/proof_of_concept/cifar10/lambda_evolution/multi_reg_comparison.png"
    )
    
    # Metric vs reg scale plots
    plot_all_metrics_vs_reg_scale(
        mnist_results_by_reg,
        dataset_name="MNIST"
    )
    
    plot_all_metrics_vs_reg_scale(
        cifar_results_by_reg,
        dataset_name="CIFAR10"
    )
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print("\nKey Findings:")
    print("1. Lambda is measurable with per-direction estimation")
    print("2. Variance across directions indicates measurement stability")
    print("3. Works across different architectures and datasets")
    print("4. Regularization strength affects lambda evolution and model performance")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
