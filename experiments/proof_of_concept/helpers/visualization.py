"""
Visualization utilities for lambda experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional


def plot_lambda_evolution(
    tracker_history: Dict[str, List],
    title: str = "Lambda Evolution During Training",
    save_path: Optional[str] = None
):
    """
    Plot lambda mean and standard deviation over training epochs.
    
    Args:
        tracker_history: Dictionary from LambdaTracker.get_history()
        title: Plot title
        save_path: Optional path to save figure
    """
    epochs = tracker_history['epochs']
    lambda_means = tracker_history['lambda_means']
    lambda_stds = tracker_history['lambda_stds']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean with error bars
    ax.errorbar(
        epochs, lambda_means, yerr=lambda_stds,
        marker='o', linestyle='-', linewidth=2,
        capsize=5, capthick=2, label='λ (mean ± std)'
    )
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Lambda (λ)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Plot saved (suppress print for cleaner output)
    else:
        # Save to results/proof_of_concept by default
        from pathlib import Path
        output_dir = Path('results/proof_of_concept')
        output_dir.mkdir(parents=True, exist_ok=True)
        default_path = output_dir / 'lambda_evolution.png'
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {default_path}")
    
    plt.close()


def plot_derivative_profile(
    orders: List[int],
    log_derivatives: List[float],
    lambda_estimate: float,
    title: str = "Derivative Growth Profile",
    save_path: Optional[str] = None
):
    """
    Plot log(|derivative|) vs order with fitted line.
    
    Args:
        orders: List of derivative orders [1, 2, 3, ...]
        log_derivatives: List of log|d_n| values
        lambda_estimate: Fitted lambda (slope)
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax.scatter(orders, log_derivatives, s=100, alpha=0.7, label='Data', color='blue')
    
    # Plot fitted line
    intercept = log_derivatives[0] - lambda_estimate * orders[0]
    fitted_line = [intercept + lambda_estimate * n for n in orders]
    ax.plot(orders, fitted_line, 'r--', linewidth=2, 
            label=f'Fit: λ = {lambda_estimate:.4f}')
    
    ax.set_xlabel('Derivative Order (n)', fontsize=12)
    ax.set_ylabel('log|d_n|', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Plot saved (suppress print for cleaner output)
    
    plt.close()


def plot_cross_dataset_comparison(
    results: Dict[str, Dict],
    metric: str = 'lambda_mean',
    title: str = "Lambda Comparison Across Datasets",
    save_path: Optional[str] = None
):
    """
    Compare lambda values across different datasets/experiments.
    
    Args:
        results: Dictionary mapping experiment names to result dictionaries
                 Each result should have 'lambda_mean' and 'lambda_std'
        metric: Which metric to plot ('lambda_mean', 'lambda_std', etc.)
        title: Plot title
        save_path: Optional path to save figure
    
    Example:
        >>> results = {
        ...     'MNIST': {'lambda_mean': 0.45, 'lambda_std': 0.12},
        ...     'CIFAR-10': {'lambda_mean': 0.62, 'lambda_std': 0.15}
        ... }
        >>> plot_cross_dataset_comparison(results)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(results.keys())
    means = [results[name]['lambda_mean'] for name in names]
    stds = [results[name]['lambda_std'] for name in names]
    
    x_pos = np.arange(len(names))
    
    # Bar plot with error bars
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, 
                   alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Dataset/Experiment', fontsize=12)
    ax.set_ylabel('Lambda (λ)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Plot saved (suppress print for cleaner output)
    else:
        # Save to results/proof_of_concept by default
        from pathlib import Path
        output_dir = Path('results/proof_of_concept')
        output_dir.mkdir(parents=True, exist_ok=True)
        default_path = output_dir / 'cross_dataset_comparison.png'
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {default_path}")
    
    plt.close()


def plot_lambda_distribution(
    lambda_values: List[float],
    title: str = "Lambda Distribution Across Directions",
    save_path: Optional[str] = None,
    reg_scale: Optional[float] = None
):
    """
    Plot histogram of lambda values from different directions.
    
    Args:
        lambda_values: List of lambda estimates (one per direction)
        title: Plot title
        save_path: Optional path to save figure
        reg_scale: Optional regularization scale to display in the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(lambda_values, bins=15, alpha=0.7, 
                                color='steelblue', edgecolor='black')
    
    # Add mean line
    mean_lambda = np.mean(lambda_values)
    ax.axvline(mean_lambda, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_lambda:.4f}')
    
    # Add std and reg_scale annotation
    std_lambda = np.std(lambda_values, ddof=1)
    annotation_text = f'Std: {std_lambda:.4f}'
    if reg_scale is not None:
        annotation_text += f'\nReg Scale: {reg_scale}'
    ax.text(0.02, 0.98, annotation_text, 
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Plot saved (suppress print for cleaner output)
    else:
        # Save to results/proof_of_concept by default
        from pathlib import Path
        output_dir = Path('results/proof_of_concept')
        output_dir.mkdir(parents=True, exist_ok=True)
        default_path = output_dir / 'lambda_distribution.png'
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {default_path}")
    
    plt.close()
