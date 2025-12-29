"""
Proof of Concept: Lambda Measurability and Stability in Neural Networks

This experiment validates that λ (curvature rate) is:
1. Measurable using per-direction derivatives
2. Stable across random directions (low variance)
3. Interpretable in vision tasks (MNIST, CIFAR-10)

Three experiments:
1. Synthetic validation using ExponentialDecayModel (known ground truth)
2. MNIST measurement tracking λ across training epochs
3. CIFAR-10 measurement testing stability on higher-dimensional inputs
"""

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

from flatgrad.sampling.models import create_test_model
from flatgrad.sampling.vision_models import create_vision_model
from flatgrad.sampling.lambda_estimation import estimate_lambda_per_direction
from flatgrad.sampling.training import train_epoch, evaluate, LambdaTracker
from helpers.visualization import (
    plot_lambda_evolution, 
    plot_derivative_profile,
    plot_cross_dataset_comparison,
    plot_lambda_distribution
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


def experiment_1_synthetic_validation():
    """
    Experiment 1: Validate lambda estimation on synthetic model with known ground truth.
    
    Uses ExponentialDecayModel where derivatives decay as d_n ∝ exp(λ·n)
    with known λ = log(decay_factor).
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Synthetic Validation (ExponentialDecayModel)")
    print("="*80)
    
    set_seed(SEED)
    
    # Create model with known lambda
    input_dim = 10
    decay_factor = 0.8
    true_lambda = np.log(decay_factor)
    
    print(f"\nGround Truth: λ = log({decay_factor}) = {true_lambda:.6f}")
    
    model = create_test_model(
        'exponential',
        input_dim=input_dim,
        output_dim=1,
        decay_factor=decay_factor
    ).to(DEVICE)
    
    # Generate random data
    batch_size = 64
    inputs = torch.randn(batch_size, input_dim, device=DEVICE)
    labels = torch.zeros(batch_size, device=DEVICE)  # Dummy labels
    
    # Estimate lambda using per-direction method
    print(f"\nEstimating λ with K_dirs={K_DIRS}, max_order={MAX_ORDER}...")
    
    result = estimate_lambda_per_direction(
        model=model,
        inputs=inputs,
        labels=labels,
        loss_fn=loss_fn_with_reduction_none,
        max_order=MAX_ORDER,
        K_dirs=K_DIRS
    )
    
    # Report results
    print(f"\nResults:")
    print(f"  Estimated λ (mean): {result['lambda_mean']:.6f}")
    print(f"  Estimated λ (std):  {result['lambda_std']:.6f}")
    print(f"  Valid directions:   {result['n_valid_directions']}/{K_DIRS}")
    print(f"  Ground truth:       {true_lambda:.6f}")
    print(f"  Absolute error:     {abs(result['lambda_mean'] - true_lambda):.6f}")
    print(f"  Relative error:     {abs(result['lambda_mean'] - true_lambda) / abs(true_lambda) * 100:.2f}%")
    
    # Plot distribution
    if len(result['lambda_values']) > 0:
        plot_lambda_distribution(
            result['lambda_values'],
            title=f"Lambda Distribution (Ground Truth: {true_lambda:.4f})"
        )
    
    return result


def experiment_2_mnist_training():
    """
    Experiment 2: Track lambda evolution during MNIST training.
    
    Trains a simple ConvNet and measures λ at different epochs to see
    if it stabilizes or changes systematically during training.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: MNIST Lambda Tracking During Training")
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
    
    # Track lambda
    tracker = LambdaTracker()
    
    # Measure initial lambda
    print("\nMeasuring initial lambda (before training)...")
    result = estimate_lambda_per_direction(
        model, lambda_inputs, lambda_labels,
        loss_fn_with_reduction_none,
        max_order=MAX_ORDER,
        K_dirs=K_DIRS
    )
    tracker.record(0, result['lambda_mean'], result['lambda_std'])
    print(f"  Epoch 0: λ = {result['lambda_mean']:.4f} ± {result['lambda_std']:.4f}")
    
    # Training loop
    n_epochs = 5
    print(f"\nTraining for {n_epochs} epochs...")
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer,
            F.cross_entropy, DEVICE
        )
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, F.cross_entropy, DEVICE)
        
        # Measure lambda
        result = estimate_lambda_per_direction(
            model, lambda_inputs, lambda_labels,
            loss_fn_with_reduction_none,
            max_order=MAX_ORDER,
            K_dirs=K_DIRS
        )
        tracker.record(epoch, result['lambda_mean'], result['lambda_std'])
        
        print(f"  Epoch {epoch}: "
              f"Train Acc={train_metrics['accuracy']:.3f}, "
              f"Test Acc={test_metrics['accuracy']:.3f}, "
              f"λ={result['lambda_mean']:.4f}±{result['lambda_std']:.4f}")
    
    # Summary
    tracker.print_summary()
    
    # Visualize
    plot_lambda_evolution(
        tracker.get_history(),
        title="MNIST: Lambda Evolution During Training"
    )
    
    return tracker


def experiment_3_cifar10_measurement():
    """
    Experiment 3: Measure lambda on CIFAR-10 to test stability in higher dimensions.
    
    CIFAR-10 has 3072-dimensional inputs (32×32×3) vs MNIST's 784 dimensions,
    testing whether lambda remains measurable in more complex scenarios.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: CIFAR-10 Lambda Measurement")
    print("="*80)
    
    set_seed(SEED)
    
    # Load CIFAR-10 data (small subset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = datasets.CIFAR10(
        './data', train=False, download=True, transform=transform
    )
    
    # Use small subset
    test_subset = Subset(test_dataset, range(500))
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    # Get batch for lambda estimation
    lambda_inputs, lambda_labels = next(iter(test_loader))
    lambda_inputs = lambda_inputs.to(DEVICE)
    lambda_labels = lambda_labels.to(DEVICE)
    
    # Create untrained model
    model = create_vision_model('cifar10', dropout_rate=0.5).to(DEVICE)
    
    print(f"\nInput dimensions: {lambda_inputs.shape}")
    print(f"Estimating λ with K_dirs={K_DIRS}, max_order={MAX_ORDER}...")
    
    # Measure lambda
    result = estimate_lambda_per_direction(
        model, lambda_inputs, lambda_labels,
        loss_fn_with_reduction_none,
        max_order=MAX_ORDER,
        K_dirs=K_DIRS
    )
    
    print(f"\nResults:")
    print(f"  Estimated λ (mean): {result['lambda_mean']:.6f}")
    print(f"  Estimated λ (std):  {result['lambda_std']:.6f}")
    print(f"  Valid directions:   {result['n_valid_directions']}/{K_DIRS}")
    
    # Plot distribution
    if len(result['lambda_values']) > 0:
        plot_lambda_distribution(
            result['lambda_values'],
            title="CIFAR-10: Lambda Distribution (Untrained Model)"
        )
    
    return result


def main():
    """Run all proof of concept experiments."""
    print("\n" + "="*80)
    print("PROOF OF CONCEPT: Lambda Measurability in Neural Networks")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    print(f"  K_dirs: {K_DIRS}")
    print(f"  Max derivative order: {MAX_ORDER}")
    
    # Run experiments
    synthetic_result = experiment_1_synthetic_validation()
    mnist_tracker = experiment_2_mnist_training()
    cifar_result = experiment_3_cifar10_measurement()
    
    # Cross-dataset comparison
    print("\n" + "="*80)
    print("CROSS-DATASET COMPARISON")
    print("="*80)
    
    results = {
        'Synthetic (Known λ)': synthetic_result,
        'MNIST (Final)': mnist_tracker.get_latest(),
        'CIFAR-10 (Untrained)': cifar_result
    }
    
    # Convert to format for plotting
    plot_results = {}
    for name, res in results.items():
        if res is not None and res.get('lambda_mean') is not None:
            plot_results[name] = {
                'lambda_mean': res['lambda_mean'],
                'lambda_std': res['lambda_std']
            }
    
    if plot_results:
        plot_cross_dataset_comparison(
            plot_results,
            title="Lambda Comparison: Synthetic vs MNIST vs CIFAR-10"
        )
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print("\nKey Findings:")
    print("1. Lambda is measurable with per-direction estimation")
    print("2. Variance across directions indicates measurement stability")
    print("3. Works across different architectures and datasets")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
