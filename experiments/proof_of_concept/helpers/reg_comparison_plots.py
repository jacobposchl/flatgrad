"""
Visualization functions for comparing different regularization strengths.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional


def plot_lambda_evolution_multi_reg(
    results_by_reg: Dict[float, Dict],
    dataset_name: str = "MNIST",
    save_path: Optional[str] = None
):
    """
    Plot lambda evolution for multiple regularization strengths.
    
    Args:
        results_by_reg: Dictionary mapping reg_scale to experiment results
                       Each result should have a 'tracker' with get_history()
        dataset_name: Name of the dataset (for title)
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort by reg_scale for consistent ordering
    reg_scales = sorted(results_by_reg.keys())
    
    # Color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(reg_scales)))
    
    for i, reg_scale in enumerate(reg_scales):
        result = results_by_reg[reg_scale]
        tracker_history = result['tracker'].get_history()
        
        epochs = tracker_history['epochs']
        lambda_means = tracker_history['lambda_means']
        lambda_stds = tracker_history['lambda_stds']
        
        # Plot mean with shaded std
        ax.plot(epochs, lambda_means, marker='o', linestyle='-', linewidth=2,
                color=colors[i], label=f'Reg Scale: {reg_scale}')
        ax.fill_between(epochs, 
                        np.array(lambda_means) - np.array(lambda_stds),
                        np.array(lambda_means) + np.array(lambda_stds),
                        alpha=0.2, color=colors[i])
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Lambda (λ)', fontsize=12)
    ax.set_title(f'{dataset_name}: Lambda Evolution for Different Regularization Strengths', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_metric_vs_reg_scale(
    reg_scales: List[float],
    metric_values: List[float],
    metric_name: str = "Accuracy",
    dataset_name: str = "MNIST",
    save_path: Optional[str] = None
):
    """
    Plot a metric (accuracy, ECE, or generalization gap) vs regularization scale.
    
    Args:
        reg_scales: List of regularization scale values
        metric_values: Corresponding metric values
        metric_name: Name of the metric (for labeling)
        dataset_name: Name of the dataset
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by reg_scale
    sorted_indices = np.argsort(reg_scales)
    reg_scales_sorted = np.array(reg_scales)[sorted_indices]
    metric_values_sorted = np.array(metric_values)[sorted_indices]
    
    # Plot
    ax.plot(reg_scales_sorted, metric_values_sorted, marker='o', linestyle='-', 
            linewidth=2, markersize=8, color='steelblue')
    
    # Add value labels
    for x, y in zip(reg_scales_sorted, metric_values_sorted):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    ax.set_xlabel('Regularization Scale', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{dataset_name}: {metric_name} vs Regularization Strength', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_all_metrics_vs_reg_scale(
    results_by_reg: Dict[float, Dict],
    dataset_name: str = "MNIST",
    save_path_prefix: Optional[str] = None
):
    """
    Create plots for accuracy, ECE, and generalization gap vs reg scale.
    
    Args:
        results_by_reg: Dictionary mapping reg_scale to experiment results
        dataset_name: Name of the dataset
        save_path_prefix: Optional prefix for save paths (will add metric name)
    """
    reg_scales = sorted(results_by_reg.keys())
    
    # Extract metrics
    test_accuracies = []
    test_eces = []
    gen_gaps = []
    
    for reg_scale in reg_scales:
        result = results_by_reg[reg_scale]
        test_accuracies.append(result['final_test']['accuracy'])
        test_eces.append(result['final_test'].get('ece', 0.0))
        gen_gap = result['final_train']['accuracy'] - result['final_test']['accuracy']
        gen_gaps.append(gen_gap)
    
    # Plot accuracy
    if save_path_prefix:
        acc_path = f"{save_path_prefix}_accuracy.png"
    else:
        acc_path = f"results/proof_of_concept/{dataset_name.lower()}/metrics_vs_reg/accuracy.png"
    plot_metric_vs_reg_scale(reg_scales, test_accuracies, "Test Accuracy", 
                            dataset_name, acc_path)
    
    # Plot ECE
    if save_path_prefix:
        ece_path = f"{save_path_prefix}_ece.png"
    else:
        ece_path = f"results/proof_of_concept/{dataset_name.lower()}/metrics_vs_reg/ece.png"
    plot_metric_vs_reg_scale(reg_scales, test_eces, "Expected Calibration Error (ECE)", 
                            dataset_name, ece_path)
    
    # Plot generalization gap
    if save_path_prefix:
        gap_path = f"{save_path_prefix}_gen_gap.png"
    else:
        gap_path = f"results/proof_of_concept/{dataset_name.lower()}/metrics_vs_reg/gen_gap.png"
    plot_metric_vs_reg_scale(reg_scales, gen_gaps, "Generalization Gap (Train - Test Acc)", 
                            dataset_name, gap_path)
