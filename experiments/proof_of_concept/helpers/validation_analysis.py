"""
Post-hoc validation analyses for proof-of-concept experiments.

Implements 4 validation analyses:
1. Direction Convergence Analysis
2. Order Sensitivity Analysis
3. Temporal Stability Analysis (Spaghetti Plots)
4. Joint K-Order Optimization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd


def load_lambda_data(npz_path: str) -> dict:
    """Load lambda data from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def recompute_lambda_from_derivatives(derivatives_per_dir: list, K: int = None, 
                                     order_start: int = 1, order_end: int = 6) -> Tuple[float, float, list]:
    """
    Recompute lambda from stored derivatives with different K and order configurations.
    
    Args:
        derivatives_per_dir: List of derivative lists (one per direction)
        K: Number of directions to use (None = use all)
        order_start: Starting derivative order (1-based)
        order_end: Ending derivative order (1-based, inclusive)
    
    Returns:
        Tuple of (lambda_mean, lambda_std, lambda_values)
    """
    if len(derivatives_per_dir) == 0:
        return None, None, []
    
    # Limit to K directions
    if K is not None:
        derivatives_per_dir = derivatives_per_dir[:K]
    
    lambda_values = []
    
    for deriv_list in derivatives_per_dir:
        if len(deriv_list) == 0:
            continue
        
        # Extract specified order range (convert to 0-indexed)
        selected_derivs = deriv_list[order_start-1:order_end]
        
        if len(selected_derivs) < 2:
            continue
        
        # Compute log derivatives
        log_derivs = []
        for d in selected_derivs:
            if d > 1e-10 and np.isfinite(d):
                log_derivs.append(np.log(d))
            else:
                break
        
        if len(log_derivs) >= 2:
            # Linear fit: log|d_n| vs n
            orders = np.arange(order_start, order_start + len(log_derivs))
            slope, intercept = np.polyfit(orders, log_derivs, 1)
            
            if np.isfinite(slope):
                lambda_values.append(slope)
    
    if len(lambda_values) == 0:
        return None, None, []
    
    lambda_array = np.array(lambda_values)
    return float(np.mean(lambda_array)), float(np.std(lambda_array, ddof=1 if len(lambda_array) > 1 else 0)), lambda_values


def direction_convergence_analysis(lambda_data_paths: List[str], output_dir: str):
    """
    Analysis A: Direction Convergence
    
    Tests how many random directions (K) are needed for stable lambda estimation.
    Subsamples K ∈ {1, 2, 3, 5, 10, 15} and plots convergence.
    
    Args:
        lambda_data_paths: List of paths to lambda_data.npz files
        output_dir: Directory to save analysis plots
    """
    print("\n" + "="*80)
    print("Direction Convergence Analysis")
    print("="*80 + "\n")
    
    output_path = Path(output_dir) / 'direction_convergence'
    output_path.mkdir(parents=True, exist_ok=True)
    
    K_values = [1, 2, 3, 5, 10, 15]
    
    for data_path in lambda_data_paths:
        data = load_lambda_data(data_path)
        
        # Extract experiment info from path
        path_parts = Path(data_path).parts
        method_name = path_parts[-2]
        dataset = path_parts[-3]
        
        print(f"Processing: {dataset}/{method_name}")
        
        lambda_values_per_epoch = data.get('lambda_values_per_epoch', None)
        derivatives_per_epoch = data.get('derivatives_per_epoch', None)
        epochs = data['epochs']
        
        if lambda_values_per_epoch is None or len(lambda_values_per_epoch) == 0:
            print(f"  No per-direction data found, skipping...")
            continue
        
        # Select epochs to analyze (0, 10, 25, 50 if available)
        analyze_epochs_idx = []
        for target_epoch in [0, 10, 25, 50]:
            if target_epoch in epochs:
                analyze_epochs_idx.append(np.where(epochs == target_epoch)[0][0])
        
        if len(analyze_epochs_idx) == 0:
            print(f"  No analysis epochs found, skipping...")
            continue
        
        # Create convergence plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for plot_idx, epoch_idx in enumerate(analyze_epochs_idx):
            if plot_idx >= 4:
                break
            
            epoch = epochs[epoch_idx]
            lambda_vals = lambda_values_per_epoch[epoch_idx]
            
            if len(lambda_vals) == 0:
                continue
            
            # Compute mean and stderr for different K values
            means = []
            stderrs = []
            
            for K in K_values:
                if K <= len(lambda_vals):
                    subset = lambda_vals[:K]
                    means.append(np.mean(subset))
                    stderrs.append(np.std(subset, ddof=1 if len(subset) > 1 else 0) / np.sqrt(len(subset)))
                else:
                    means.append(np.nan)
                    stderrs.append(np.nan)
            
            # Plot
            ax = axes[plot_idx]
            valid_idx = ~np.isnan(means)
            ax.errorbar(np.array(K_values)[valid_idx], np.array(means)[valid_idx], 
                       yerr=np.array(stderrs)[valid_idx], marker='o', capsize=5)
            ax.set_xlabel('Number of Directions (K)')
            ax.set_ylabel('Lambda Mean ± SE')
            ax.set_title(f'Epoch {epoch}')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(K_values)
        
        plt.suptitle(f'Direction Convergence: {dataset}/{method_name}')
        plt.tight_layout()
        plt.savefig(output_path / f'{dataset}_{method_name}_convergence.png', dpi=150)
        plt.close()
        
        print(f"  Saved convergence plot")


def order_sensitivity_analysis(lambda_data_paths: List[str], output_dir: str):
    """
    Analysis B: Derivative Order Sensitivity
    
    Tests how many derivative orders are needed for reliable lambda estimation.
    Recomputes lambda using different order subsets: 1-2, 1-3, 1-4, 2-4, 2-5, 1-6.
    
    Args:
        lambda_data_paths: List of paths to lambda_data.npz files
        output_dir: Directory to save analysis plots
    """
    print("\n" + "="*80)
    print("Order Sensitivity Analysis")
    print("="*80 + "\n")
    
    output_path = Path(output_dir) / 'order_sensitivity'
    output_path.mkdir(parents=True, exist_ok=True)
    
    order_configs = [
        (1, 2, '1-2'),
        (1, 3, '1-3'),
        (1, 4, '1-4'),
        (2, 4, '2-4'),
        (2, 5, '2-5'),
        (1, 6, '1-6')
    ]
    
    for data_path in lambda_data_paths:
        data = load_lambda_data(data_path)
        
        # Extract experiment info
        path_parts = Path(data_path).parts
        method_name = path_parts[-2]
        dataset = path_parts[-3]
        
        print(f"Processing: {dataset}/{method_name}")
        
        derivatives_per_epoch = data.get('derivatives_per_epoch', None)
        epochs = data['epochs']
        
        if derivatives_per_epoch is None or len(derivatives_per_epoch) == 0:
            print(f"  No derivative data found, skipping...")
            continue
        
        # Use final epoch for comparison
        final_epoch_idx = -1
        final_derivatives = derivatives_per_epoch[final_epoch_idx]
        
        if len(final_derivatives) == 0:
            print(f"  No derivatives at final epoch, skipping...")
            continue
        
        # Compute lambda for each order configuration
        results = []
        for order_start, order_end, label in order_configs:
            lambda_mean, lambda_std, lambda_vals = recompute_lambda_from_derivatives(
                final_derivatives, K=None, order_start=order_start, order_end=order_end
            )
            
            if lambda_mean is not None:
                results.append({
                    'config': label,
                    'lambda_mean': lambda_mean,
                    'lambda_std': lambda_std,
                    'stderr': lambda_std / np.sqrt(len(lambda_vals)) if len(lambda_vals) > 0 else 0
                })
        
        if len(results) == 0:
            print(f"  No valid results, skipping...")
            continue
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        configs = [r['config'] for r in results]
        means = [r['lambda_mean'] for r in results]
        stderrs = [r['stderr'] for r in results]
        
        ax.bar(configs, means, yerr=stderrs, capsize=5, alpha=0.7)
        ax.set_xlabel('Order Configuration')
        ax.set_ylabel('Lambda Estimate')
        ax.set_title(f'Order Sensitivity: {dataset}/{method_name}')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{dataset}_{method_name}_order_sensitivity.png', dpi=150)
        plt.close()
        
        print(f"  Saved order sensitivity plot")


def temporal_stability_analysis(lambda_data_paths: List[str], output_dir: str):
    """
    Analysis C: Temporal and Directional Stability (Spaghetti Plots)
    
    Plots all individual direction lambda trajectories over training.
    Shows whether directions evolve in parallel or diverge.
    
    Args:
        lambda_data_paths: List of paths to lambda_data.npz files
        output_dir: Directory to save analysis plots
    """
    print("\n" + "="*80)
    print("Temporal Stability Analysis")
    print("="*80 + "\n")
    
    output_path = Path(output_dir) / 'temporal_stability'
    output_path.mkdir(parents=True, exist_ok=True)
    
    for data_path in lambda_data_paths:
        data = load_lambda_data(data_path)
        
        # Extract experiment info
        path_parts = Path(data_path).parts
        method_name = path_parts[-2]
        dataset = path_parts[-3]
        
        print(f"Processing: {dataset}/{method_name}")
        
        lambda_values_per_epoch = data.get('lambda_values_per_epoch', None)
        epochs = data['epochs']
        
        if lambda_values_per_epoch is None or len(lambda_values_per_epoch) == 0:
            print(f"  No per-direction data found, skipping...")
            continue
        
        # Build trajectory matrix: [n_epochs, n_directions]
        # Find max number of directions across all epochs
        max_dirs = max(len(vals) for vals in lambda_values_per_epoch if len(vals) > 0)
        
        if max_dirs == 0:
            print(f"  No valid directions, skipping...")
            continue
        
        # Create matrix with NaN for missing values
        trajectory_matrix = np.full((len(epochs), max_dirs), np.nan)
        
        for epoch_idx, lambda_vals in enumerate(lambda_values_per_epoch):
            for dir_idx, val in enumerate(lambda_vals):
                if dir_idx < max_dirs:
                    trajectory_matrix[epoch_idx, dir_idx] = val
        
        # Create spaghetti plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Spaghetti plot with all directions
        ax = axes[0]
        for dir_idx in range(max_dirs):
            traj = trajectory_matrix[:, dir_idx]
            valid_mask = ~np.isnan(traj)
            if valid_mask.sum() > 0:
                ax.plot(epochs[valid_mask], traj[valid_mask], alpha=0.3, linewidth=1, color='blue')
        
        # Add mean line
        mean_trajectory = np.nanmean(trajectory_matrix, axis=1)
        ax.plot(epochs, mean_trajectory, color='black', linewidth=3, label='Mean')
        
        # Add confidence interval
        std_trajectory = np.nanstd(trajectory_matrix, axis=1)
        ax.fill_between(epochs, mean_trajectory - std_trajectory, mean_trajectory + std_trajectory,
                        alpha=0.2, color='black', label='±1 Std')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Lambda')
        ax.set_title(f'Lambda Evolution: {dataset}/{method_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot 2: Coefficient of variation over time
        ax = axes[1]
        cv_trajectory = std_trajectory / np.abs(mean_trajectory)
        cv_trajectory[~np.isfinite(cv_trajectory)] = np.nan
        
        ax.plot(epochs, cv_trajectory, color='red', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Lambda Stability (CV = std/|mean|)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'{dataset}_{method_name}_spaghetti.png', dpi=150)
        plt.close()
        
        print(f"  Saved spaghetti plot")
        
        # Create violin plots at key epochs
        analyze_epochs = [0, 10, 25, 50]
        epoch_indices = []
        epoch_labels = []
        
        for target_epoch in analyze_epochs:
            if target_epoch in epochs:
                epoch_indices.append(np.where(epochs == target_epoch)[0][0])
                epoch_labels.append(f'Epoch {target_epoch}')
        
        if len(epoch_indices) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            violin_data = []
            for idx in epoch_indices:
                vals = lambda_values_per_epoch[idx]
                if len(vals) > 0:
                    violin_data.append(vals)
                else:
                    violin_data.append([np.nan])
            
            parts = ax.violinplot(violin_data, positions=range(len(epoch_labels)), 
                                 showmeans=True, showmedians=True)
            
            ax.set_xticks(range(len(epoch_labels)))
            ax.set_xticklabels(epoch_labels)
            ax.set_ylabel('Lambda Value')
            ax.set_title(f'Lambda Distribution Evolution: {dataset}/{method_name}')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / f'{dataset}_{method_name}_violin.png', dpi=150)
            plt.close()
            
            print(f"  Saved violin plot")


def joint_K_order_optimization(lambda_data_paths: List[str], output_dir: str):
    """
    Analysis D: Joint K-Order Optimization
    
    Tests all combinations of K ∈ {1,2,3,5,10,15} and order configs to find optimal measurement settings.
    Creates heatmaps showing lambda estimates, standard errors, and deviations from "ground truth".
    
    Args:
        lambda_data_paths: List of paths to lambda_data.npz files
        output_dir: Directory to save analysis plots
    """
    print("\n" + "="*80)
    print("Joint K-Order Optimization Analysis")
    print("="*80 + "\n")
    
    output_path = Path(output_dir) / 'joint_optimization'
    output_path.mkdir(parents=True, exist_ok=True)
    
    K_values = [1, 2, 3, 5, 10, 15]
    order_configs = [
        (1, 2, '1-2'),
        (1, 3, '1-3'),
        (1, 4, '1-4'),
        (2, 4, '2-4'),
        (2, 5, '2-5'),
        (1, 6, '1-6')
    ]
    
    for data_path in lambda_data_paths:
        data = load_lambda_data(data_path)
        
        # Extract experiment info
        path_parts = Path(data_path).parts
        method_name = path_parts[-2]
        dataset = path_parts[-3]
        
        print(f"Processing: {dataset}/{method_name}")
        
        derivatives_per_epoch = data.get('derivatives_per_epoch', None)
        epochs = data['epochs']
        
        if derivatives_per_epoch is None or len(derivatives_per_epoch) == 0:
            print(f"  No derivative data found, skipping...")
            continue
        
        # Use final epoch
        final_derivatives = derivatives_per_epoch[-1]
        
        if len(final_derivatives) == 0:
            print(f"  No derivatives at final epoch, skipping...")
            continue
        
        # Compute "ground truth" (K=15, orders 1-6)
        lambda_true, _, _ = recompute_lambda_from_derivatives(
            final_derivatives, K=None, order_start=1, order_end=6
        )
        
        if lambda_true is None:
            print(f"  Could not compute ground truth, skipping...")
            continue
        
        # Compute grid
        lambda_estimates = np.full((len(K_values), len(order_configs)), np.nan)
        std_errors = np.full((len(K_values), len(order_configs)), np.nan)
        deviations = np.full((len(K_values), len(order_configs)), np.nan)
        cvs = np.full((len(K_values), len(order_configs)), np.nan)
        
        for k_idx, K in enumerate(K_values):
            for o_idx, (order_start, order_end, label) in enumerate(order_configs):
                lambda_mean, lambda_std, lambda_vals = recompute_lambda_from_derivatives(
                    final_derivatives, K=K, order_start=order_start, order_end=order_end
                )
                
                if lambda_mean is not None:
                    lambda_estimates[k_idx, o_idx] = lambda_mean
                    std_errors[k_idx, o_idx] = lambda_std / np.sqrt(len(lambda_vals)) if len(lambda_vals) > 0 else 0
                    deviations[k_idx, o_idx] = abs(lambda_mean - lambda_true)
                    cvs[k_idx, o_idx] = abs(lambda_std / lambda_mean) if lambda_mean != 0 else np.nan
        
        # Create heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        order_labels = [cfg[2] for cfg in order_configs]
        
        # Heatmap 1: Lambda estimates
        ax = axes[0, 0]
        sns.heatmap(lambda_estimates, annot=True, fmt='.3f', cmap='viridis', ax=ax,
                   xticklabels=order_labels, yticklabels=K_values, cbar_kws={'label': 'Lambda'})
        ax.set_xlabel('Order Configuration')
        ax.set_ylabel('Number of Directions (K)')
        ax.set_title('Lambda Estimates')
        
        # Heatmap 2: Standard error
        ax = axes[0, 1]
        sns.heatmap(std_errors, annot=True, fmt='.4f', cmap='Reds', ax=ax,
                   xticklabels=order_labels, yticklabels=K_values, cbar_kws={'label': 'Std Error'})
        ax.set_xlabel('Order Configuration')
        ax.set_ylabel('Number of Directions (K)')
        ax.set_title('Standard Error')
        
        # Heatmap 3: Deviation from ground truth
        ax = axes[1, 0]
        sns.heatmap(deviations, annot=True, fmt='.4f', cmap='Oranges', ax=ax,
                   xticklabels=order_labels, yticklabels=K_values, cbar_kws={'label': 'Deviation'})
        ax.set_xlabel('Order Configuration')
        ax.set_ylabel('Number of Directions (K)')
        ax.set_title(f'|λ - λ_true| (true={lambda_true:.3f})')
        
        # Heatmap 4: Coefficient of variation
        ax = axes[1, 1]
        sns.heatmap(cvs, annot=True, fmt='.3f', cmap='Blues', ax=ax,
                   xticklabels=order_labels, yticklabels=K_values, cbar_kws={'label': 'CV'})
        ax.set_xlabel('Order Configuration')
        ax.set_ylabel('Number of Directions (K)')
        ax.set_title('Coefficient of Variation')
        
        plt.suptitle(f'Joint K-Order Optimization: {dataset}/{method_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / f'{dataset}_{method_name}_joint_optimization.png', dpi=150)
        plt.close()
        
        print(f"  Saved joint optimization heatmaps")


def run_all_validation_analyses(results_dir: str, output_dir: str = None):
    """
    Run all 4 validation analyses on experiment results.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save analysis outputs (default: results_dir/validation_analyses)
    """
    if output_dir is None:
        output_dir = str(Path(results_dir) / 'validation_analyses')
    
    # Find all lambda_data.npz files
    lambda_data_paths = list(Path(results_dir).rglob('lambda_data.npz'))
    
    if len(lambda_data_paths) == 0:
        print("No lambda data files found!")
        return
    
    print(f"\nFound {len(lambda_data_paths)} experiment results")
    print(f"Output directory: {output_dir}\n")
    
    # Run analyses
    direction_convergence_analysis(lambda_data_paths, output_dir)
    order_sensitivity_analysis(lambda_data_paths, output_dir)
    temporal_stability_analysis(lambda_data_paths, output_dir)
    joint_K_order_optimization(lambda_data_paths, output_dir)
    
    print("\n" + "="*80)
    print("All validation analyses complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")
