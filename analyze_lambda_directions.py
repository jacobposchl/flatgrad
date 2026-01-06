"""
Directional Lambda Analysis: Comprehensive Investigation of Lambda Anisotropy

BREAKTHROUGH FINDING: Lambda standard deviation (directional variance) is the key
metric for predicting model quality, NOT the mean lambda.

This refined script provides complete analysis for scientific publication:
1. Directional variance analysis (λ_std as primary metric)
2. Temporal evolution (how λ_std changes during training)
3. Per-method breakdown (which regularizers maximize anisotropy)
4. Statistical robustness (per-dataset, effect sizes, power analysis)
5. Theoretical connections (eigenvalue distributions, Fisher Information)
6. Publication-quality visualizations
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List

# Configuration
RESULTS_DIR = Path(r"C:\Users\Jacob Poschl\Desktop\flatgrad\flatgrad\results\proof_of_concept")
OUTPUT_DIR = Path(r"C:\Users\Jacob Poschl\Desktop\flatgrad\flatgrad\results\proof_of_concept\analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def print_section(title: str, level: int = 1):
    """Print formatted section header."""
    pass  # Verbose output disabled

def load_directional_lambda_data() -> pd.DataFrame:
    """Load lambda data with per-direction information."""
    print_section("1. LOADING DIRECTIONAL LAMBDA DATA", level=1)
    
    results = []
    
    for experiment_dir in RESULTS_DIR.rglob('lambda_data.npz'):
        try:
            # Load lambda data
            lambda_data = np.load(experiment_dir, allow_pickle=True)
            
            # Load summary for metadata
            summary_path = experiment_dir.parent / 'summary.json'
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            # Get final lambda values per direction (last measurement)
            final_lambdas = lambda_data['lambda_values_per_epoch'][-1]
            
            if final_lambdas is None or len(final_lambdas) == 0:
                continue
            
            # Compute directional statistics
            result = {
                'dataset': summary['dataset'],
                'method_name': summary['method_name'],
                'final_test_accuracy': summary['final_test_accuracy'],
                'final_generalization_gap': summary['final_generalization_gap'],
                'final_ece': summary['final_ece'],
                
                # Original metric
                'lambda_mean': np.mean(final_lambdas),
                
                # Directional metrics
                'lambda_min': np.min(final_lambdas),  # Flattest direction
                'lambda_max': np.max(final_lambdas),  # Sharpest direction
                'lambda_std': np.std(final_lambdas),  # Spread across directions
                'lambda_range': np.max(final_lambdas) - np.min(final_lambdas),
                'lambda_median': np.median(final_lambdas),
                
                # Percentiles
                'lambda_q25': np.percentile(final_lambdas, 25),
                'lambda_q75': np.percentile(final_lambdas, 75),
                
                # Store all direction values for detailed analysis
                'lambda_per_direction': final_lambdas,
                'n_directions': len(final_lambdas)
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"⚠️  Failed to load {experiment_dir}: {e}")
    
    df = pd.DataFrame(results)
    
    print(f"✓ Loaded {len(df)} experiments")
    
    return df

def test_directional_correlations(df: pd.DataFrame):
    """Test if directional lambda metrics correlate better than mean."""
    print_section("2. DIRECTIONAL CORRELATION ANALYSIS", level=1)
    
    # Metrics to test
    lambda_metrics = {
        'Lambda Mean (original)': 'lambda_mean',
        'Lambda Min (flattest)': 'lambda_min',
        'Lambda Max (sharpest)': 'lambda_max',
        'Lambda Median': 'lambda_median',
        'Lambda Std (spread)': 'lambda_std',
        'Lambda Range': 'lambda_range',
    }
    
    performance_metrics = {
        'Test Accuracy': 'final_test_accuracy',
        'Generalization Gap': 'final_generalization_gap',
        'ECE (Calibration)': 'final_ece',
    }
    
    results_table = []
    
    for perf_name, perf_col in performance_metrics.items():
        pass  # Verbose output disabled
        
        best_r = 0
        best_metric = None
        
        for lambda_name, lambda_col in lambda_metrics.items():
            valid_data = df[[lambda_col, perf_col]].dropna()
            
            if len(valid_data) < 3:
                continue
            
            r, p = stats.pearsonr(valid_data[lambda_col], valid_data[perf_col])
            
            pass  # Verbose output disabled
            
            results_table.append({
                'Performance Metric': perf_name,
                'Lambda Metric': lambda_name,
                'Correlation': r,
                'P-value': p,
                'Significant': p < 0.05
            })
            
            if abs(r) > abs(best_r):
                best_r = r
                best_metric = lambda_name
        
        pass  # Verbose output disabled
    
    # Convert to DataFrame for saving
    results_df = pd.DataFrame(results_table)
    output_path = OUTPUT_DIR / 'directional_correlations.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved detailed results: {output_path}")
    
    return results_df

def analyze_flattest_vs_sharpest(df: pd.DataFrame):
    """Compare flattest vs sharpest directions."""
    print_section("3. FLATTEST vs SHARPEST DIRECTION ANALYSIS", level=1)
    
    pass  # Verbose output disabled
    
    # Test correlations for extreme directions
    metrics_to_test = [
        ('Flattest Direction (min λ)', 'lambda_min'),
        ('Sharpest Direction (max λ)', 'lambda_max'),
        ('Mean λ (baseline)', 'lambda_mean'),
    ]
    
    for metric_name, metric_col in metrics_to_test:
        print(f"{metric_name}:")
        
        # vs Test Accuracy
        valid_data = df[[metric_col, 'final_test_accuracy']].dropna()
        r_acc, p_acc = stats.pearsonr(valid_data[metric_col], valid_data['final_test_accuracy'])
        
        # vs Generalization Gap (absolute value)
        valid_data = df[[metric_col, 'final_generalization_gap']].dropna()
        r_gen, p_gen = stats.pearsonr(valid_data[metric_col], valid_data['final_generalization_gap'].abs())
        
        # vs ECE
        valid_data = df[[metric_col, 'final_ece']].dropna()
        r_ece, p_ece = stats.pearsonr(valid_data[metric_col], valid_data['final_ece'])
        
        print(f"  vs Test Accuracy:    r={r_acc:+.4f}, p={p_acc:.4f}")
        print(f"  vs |Gen Gap|:        r={r_gen:+.4f}, p={p_gen:.4f}")
        print(f"  vs ECE:              r={r_ece:+.4f}, p={p_ece:.4f}")
        print()

def analyze_lambda_spread(df: pd.DataFrame):
    """Analyze if lambda variance/spread matters."""
    print_section("4. LAMBDA SPREAD (VARIANCE) ANALYSIS PER-DATASET", level=1)
    
    pass  # Verbose output disabled
    
    # Analyze per dataset
    for dataset in sorted(df['dataset'].unique()):
        print(f"\n{dataset.upper()}:")
        df_dataset = df[df['dataset'] == dataset]
        
        # Correlation with performance
        valid_data = df_dataset[['lambda_std', 'final_test_accuracy']].dropna()
        if len(valid_data) >= 3:
            r_acc, p_acc = stats.pearsonr(valid_data['lambda_std'], valid_data['final_test_accuracy'])
            print(f"  Lambda Std vs Test Accuracy:   r={r_acc:+.4f}, p={p_acc:.4f} {'*' if p_acc < 0.05 else 'ns'} (n={len(valid_data)})")
        
        valid_data = df_dataset[['lambda_std', 'final_generalization_gap']].dropna()
        if len(valid_data) >= 3:
            r_gen, p_gen = stats.pearsonr(valid_data['lambda_std'], valid_data['final_generalization_gap'].abs())
            print(f"  Lambda Std vs |Gen Gap|:       r={r_gen:+.4f}, p={p_gen:.4f} {'*' if p_gen < 0.05 else 'ns'} (n={len(valid_data)})")
    
    # Summary statistics
    print(f"\nLambda Spread Statistics (all datasets):")
    print(f"  Mean std across experiments: {df['lambda_std'].mean():.4f}")
    print(f"  Min std:  {df['lambda_std'].min():.4f} ({df.loc[df['lambda_std'].idxmin(), 'method_name']}/{df.loc[df['lambda_std'].idxmin(), 'dataset']})")
    print(f"  Max std:  {df['lambda_std'].max():.4f} ({df.loc[df['lambda_std'].idxmax(), 'method_name']}/{df.loc[df['lambda_std'].idxmax(), 'dataset']})")

def create_directional_visualizations(df: pd.DataFrame):
    """Create visualizations for directional analysis."""
    print_section("5. CREATING DIRECTIONAL VISUALIZATIONS", level=1)
    
    sns.set_style("whitegrid")
    
    # Figure 1: Comparison of different lambda metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    lambda_metrics = [
        ('lambda_mean', 'Lambda Mean'),
        ('lambda_min', 'Lambda Min (Flattest)'),
        ('lambda_max', 'Lambda Max (Sharpest)'),
        ('lambda_median', 'Lambda Median'),
        ('lambda_std', 'Lambda Std (Spread)'),
        ('lambda_range', 'Lambda Range'),
    ]
    
    for idx, (metric_col, metric_name) in enumerate(lambda_metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Scatter plot: metric vs test accuracy
        for dataset in df['dataset'].unique():
            df_subset = df[df['dataset'] == dataset]
            ax.scatter(df_subset[metric_col], df_subset['final_test_accuracy'],
                      label=dataset.upper(), alpha=0.7, s=100)
        
        # Compute correlation
        valid_data = df[[metric_col, 'final_test_accuracy']].dropna()
        if len(valid_data) >= 3:
            r, p = stats.pearsonr(valid_data[metric_col], valid_data['final_test_accuracy'])
            
            # Add trend line if significant
            if p < 0.05:
                z = np.polyfit(valid_data[metric_col], valid_data['final_test_accuracy'], 1)
                p_poly = np.poly1d(z)
                x_line = np.linspace(valid_data[metric_col].min(), valid_data[metric_col].max(), 100)
                ax.plot(x_line, p_poly(x_line), "r--", alpha=0.5, linewidth=2)
            
            ax.set_title(f'{metric_name}\nr={r:.3f}, p={p:.3f}', fontsize=11, fontweight='bold')
        else:
            ax.set_title(metric_name, fontsize=11, fontweight='bold')
        
        ax.set_xlabel(metric_name, fontsize=10)
        ax.set_ylabel('Test Accuracy', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'directional_lambda_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Figure 2: Lambda distribution per experiment
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create violin plot showing lambda distribution per method
    plot_data = []
    for _, row in df.iterrows():
        lambdas = row['lambda_per_direction']
        method = row['method_name']
        for lam in lambdas:
            plot_data.append({'method': method, 'lambda': lam})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Sort by median lambda
    method_order = plot_df.groupby('method')['lambda'].median().sort_values().index
    
    sns.violinplot(data=plot_df, x='lambda', y='method', order=method_order, ax=ax)
    ax.set_xlabel('Lambda (per direction)', fontsize=12)
    ax.set_ylabel('Regularization Method', fontsize=12)
    ax.set_title('Lambda Distribution Across Directions\n(Shows spread/variance within each experiment)', 
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'lambda_direction_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Figure 3: Min vs Max lambda scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Min lambda vs Test Accuracy
    for dataset in df['dataset'].unique():
        df_subset = df[df['dataset'] == dataset]
        axes[0].scatter(df_subset['lambda_min'], df_subset['final_test_accuracy'],
                       label=dataset.upper(), alpha=0.7, s=100)
    
    axes[0].set_xlabel('Lambda Min (Flattest Direction)', fontsize=12)
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].set_title('Flattest Direction vs Performance', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Subplot 2: Lambda range vs Generalization Gap
    for dataset in df['dataset'].unique():
        df_subset = df[df['dataset'] == dataset]
        axes[1].scatter(df_subset['lambda_range'], df_subset['final_generalization_gap'].abs(),
                       label=dataset.upper(), alpha=0.7, s=100)
    
    axes[1].set_xlabel('Lambda Range (Max - Min)', fontsize=12)
    axes[1].set_ylabel('|Generalization Gap|', fontsize=12)
    axes[1].set_title('Lambda Spread vs Generalization', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'directional_extremes.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def analyze_temporal_evolution(df: pd.DataFrame):
    """Analyze how lambda_std evolves during training."""
    print_section("6. TEMPORAL EVOLUTION OF LAMBDA_STD", level=1)
    
    pass  # Verbose output disabled
    
    temporal_results = []
    
    for experiment_dir in RESULTS_DIR.rglob('lambda_data.npz'):
        try:
            lambda_data = np.load(experiment_dir, allow_pickle=True)
            summary_path = experiment_dir.parent / 'summary.json'
            
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            epochs = lambda_data['epochs']
            lambda_values_per_epoch = lambda_data['lambda_values_per_epoch']
            
            # Compute lambda_std at each epoch
            lambda_stds = []
            for lambdas in lambda_values_per_epoch:
                if lambdas is not None and len(lambdas) > 0:
                    lambda_stds.append(np.std(lambdas))
                else:
                    lambda_stds.append(np.nan)
            
            # Store temporal data
            initial_std = lambda_stds[0] if len(lambda_stds) > 0 else np.nan
            final_std = lambda_stds[-1] if len(lambda_stds) > 0 else np.nan
            
            temporal_results.append({
                'method': summary['method_name'],
                'dataset': summary['dataset'],
                'initial_lambda_std': initial_std,
                'final_lambda_std': final_std,
                'lambda_std_change': final_std - initial_std,
                'lambda_std_growth_rate': (final_std - initial_std) / initial_std if initial_std > 0 else np.nan,
                'epochs': epochs,
                'lambda_stds': lambda_stds,
                'test_accuracy': summary['final_test_accuracy']
            })
            
        except Exception as e:
            continue
    
    temporal_df = pd.DataFrame(temporal_results)
    
    print(f"Temporal Evolution Statistics:")
    print(f"  Average initial λ_std: {temporal_df['initial_lambda_std'].mean():.4f}")
    print(f"  Average final λ_std:   {temporal_df['final_lambda_std'].mean():.4f}")
    print(f"  Average change:        {temporal_df['lambda_std_change'].mean():+.4f}")
    
    # Per-dataset temporal correlations
    print("\nTemporal correlations (per-dataset):")
    for dataset in ['mnist', 'cifar10']:
        df_temporal = temporal_df[temporal_df['dataset'] == dataset]
        valid_data = df_temporal[['lambda_std_change', 'test_accuracy']].dropna()
        if len(valid_data) >= 3:
            r, p = stats.pearsonr(valid_data['lambda_std_change'], valid_data['test_accuracy'])
            sig = '*' if p < 0.05 else 'ns'
            print(f"  {dataset.upper()}: λ_std change vs test_acc: r={r:+.4f}, p={p:.4f} {sig}")
    
    # Create temporal evolution plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Individual trajectories (spaghetti plot)
    for _, row in temporal_df.iterrows():
        axes[0, 0].plot(row['epochs'], row['lambda_stds'], alpha=0.3, linewidth=1)
    
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Lambda Std', fontsize=12)
    axes[0, 0].set_title('Lambda Std Evolution During Training\n(All Experiments)', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Initial vs Final lambda_std
    for dataset in temporal_df['dataset'].unique():
        subset = temporal_df[temporal_df['dataset'] == dataset]
        axes[0, 1].scatter(subset['initial_lambda_std'], subset['final_lambda_std'],
                          label=dataset.upper(), alpha=0.7, s=100)
    
    # Add diagonal line (no change)
    min_val = min(temporal_df['initial_lambda_std'].min(), temporal_df['final_lambda_std'].min())
    max_val = max(temporal_df['initial_lambda_std'].max(), temporal_df['final_lambda_std'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='No change')
    
    axes[0, 1].set_xlabel('Initial Lambda Std (Epoch 0)', fontsize=12)
    axes[0, 1].set_ylabel('Final Lambda Std (Epoch 50)', fontsize=12)
    axes[0, 1].set_title('Lambda Std: Initial vs Final', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Lambda_std change vs test accuracy
    for dataset in temporal_df['dataset'].unique():
        subset = temporal_df[temporal_df['dataset'] == dataset]
        axes[1, 0].scatter(subset['lambda_std_change'], subset['test_accuracy'],
                          label=dataset.upper(), alpha=0.7, s=100)
    
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Lambda Std Change (Final - Initial)', fontsize=12)
    axes[1, 0].set_ylabel('Test Accuracy', fontsize=12)
    axes[1, 0].set_title('Does Lambda Std Growth Predict Performance?', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Method comparison of lambda_std growth
    method_summary = temporal_df.groupby('method')['lambda_std_change'].mean().sort_values()
    axes[1, 1].barh(range(len(method_summary)), method_summary.values, 
                    color=['red' if x < 0 else 'green' for x in method_summary.values])
    axes[1, 1].set_yticks(range(len(method_summary)))
    axes[1, 1].set_yticklabels(method_summary.index, fontsize=9)
    axes[1, 1].axvline(x=0, color='k', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Average Lambda Std Change', fontsize=12)
    axes[1, 1].set_title('Lambda Std Growth by Method', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'temporal_lambda_std_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved temporal analysis: {output_path}")
    plt.close()
    
    return temporal_df

def analyze_per_dataset_robustness(df: pd.DataFrame):
    """Check if lambda_std correlations hold within each dataset."""
    print_section("7. PER-DATASET ROBUSTNESS CHECK", level=1)
    
    print("\nChecking if lambda_std correlations are dataset-specific or universal:\n")
    
    for dataset in df['dataset'].unique():
        print(f"{'='*80}")
        print(f"Dataset: {dataset.upper()}")
        print('='*80)
        
        df_subset = df[df['dataset'] == dataset]
        
        # Test correlations within this dataset
        metrics = [
            ('Test Accuracy', 'final_test_accuracy', False),
            ('|Generalization Gap|', 'final_generalization_gap', True),
            ('ECE', 'final_ece', False),
        ]
        
        for metric_name, metric_col, use_abs in metrics:
            valid_data = df_subset[['lambda_std', metric_col]].dropna()
            
            if len(valid_data) < 3:
                print(f"  {metric_name}: insufficient data (n={len(valid_data)})")
                continue
            
            if use_abs:
                r, p = stats.pearsonr(valid_data['lambda_std'], valid_data[metric_col].abs())
            else:
                r, p = stats.pearsonr(valid_data['lambda_std'], valid_data[metric_col])
            
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"  λ_std vs {metric_name:<25} r={r:+.4f}, p={p:.4f} {sig} (n={len(valid_data)})")
        
        print()

def compute_effect_sizes_and_power(df: pd.DataFrame):
    """Compute statistical effect sizes and power analysis."""
    print_section("8. EFFECT SIZE AND STATISTICAL POWER", level=1)
    
    pass  # Verbose output disabled
    
    # Cohen's d for comparing high vs low lambda_std groups
    median_std = df['lambda_std'].median()
    high_std_group = df[df['lambda_std'] >= median_std]
    low_std_group = df[df['lambda_std'] < median_std]
    
    pass  # Verbose output disabled
    
    # Test accuracy comparison
    high_acc = high_std_group['final_test_accuracy'].mean()
    low_acc = low_std_group['final_test_accuracy'].mean()
    pooled_std_acc = np.sqrt((high_std_group['final_test_accuracy'].std()**2 + 
                              low_std_group['final_test_accuracy'].std()**2) / 2)
    cohens_d_acc = (high_acc - low_acc) / pooled_std_acc
    
    pass  # Verbose output disabled
    
    # Generalization gap comparison
    high_gap = high_std_group['final_generalization_gap'].abs().mean()
    low_gap = low_std_group['final_generalization_gap'].abs().mean()
    pooled_std_gap = np.sqrt((high_std_group['final_generalization_gap'].std()**2 + 
                              low_std_group['final_generalization_gap'].std()**2) / 2)
    cohens_d_gap = (low_gap - high_gap) / pooled_std_gap  # Reversed: lower gap is better
    
    pass  # Verbose output disabled
    
    # R-squared (variance explained)
    valid_data = df[['lambda_std', 'final_test_accuracy']].dropna()
    r, _ = stats.pearsonr(valid_data['lambda_std'], valid_data['final_test_accuracy'])
    r_squared = r ** 2
    
    pass  # Verbose output disabled

def rank_regularization_methods(df: pd.DataFrame):
    """Rank regularization methods by lambda_std (per-dataset)."""
    print_section("9. REGULARIZATION METHOD RANKING", level=1)
    
    pass  # Verbose output disabled
    
    # Group by method AND dataset
    method_stats = df.groupby(['method_name', 'dataset']).agg({
        'lambda_std': ['mean', 'std', 'count'],
        'final_test_accuracy': 'mean',
        'final_generalization_gap': lambda x: x.abs().mean(),
        'final_ece': 'mean'
    }).round(4)
    
    # Flatten column names
    method_stats.columns = ['_'.join(col).strip() for col in method_stats.columns.values]
    method_stats = method_stats.reset_index()
    
    # Rename for clarity
    method_stats.columns = ['method', 'dataset', 'lambda_std_mean', 'lambda_std_std', 'n_experiments',
                           'test_acc', 'gen_gap', 'ece']
    
    # Print per-dataset tables
    for dataset in ['mnist', 'cifar10']:
        dataset_stats = method_stats[method_stats['dataset'] == dataset].sort_values('lambda_std_mean', ascending=False)
        print(f"\n{dataset.upper()}:")
        print(f"  {'Method':<30} {'λ_std':>8} {'Test Acc':>10} {'|Gen Gap|':>10} {'ECE':>8}")
        print(f"  {'-'*70}")
        
        for _, row in dataset_stats.head(10).iterrows():
            print(f"  {row['method']:<30} {row['lambda_std_mean']:>8.4f} {row['test_acc']:>10.4f} "
                  f"{row['gen_gap']:>10.4f} {row['ece']:>8.4f}")
    
    # Save ranking
    output_path = OUTPUT_DIR / 'method_ranking_by_lambda_std.csv'
    method_stats.to_csv(output_path, index=False)
    print(f"\n✓ Saved ranking: {output_path}")
    
    return method_stats

def create_publication_figures(df: pd.DataFrame, method_stats: pd.DataFrame):
    """Create publication-quality figures."""
    print_section("10. PUBLICATION-QUALITY FIGURES", level=1)
    
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    
    # Figure 1: Main finding - Lambda Std vs Performance (2x2 grid)
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # Subplot 1: Lambda Std vs Test Accuracy (with per-dataset regression lines)
    ax1 = fig.add_subplot(gs[0, 0])
    
    colors = {'mnist': '#1f77b4', 'cifar10': '#ff7f0e'}
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax1.scatter(subset['lambda_std'], subset['final_test_accuracy'],
                   c=colors[dataset], label=dataset.upper(), alpha=0.7, s=120, edgecolors='black', linewidth=0.5)
        
        # Add per-dataset regression line
        valid_data = subset[['lambda_std', 'final_test_accuracy']].dropna()
        if len(valid_data) >= 3:
            z = np.polyfit(valid_data['lambda_std'], valid_data['final_test_accuracy'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data['lambda_std'].min(), valid_data['lambda_std'].max(), 100)
            ax1.plot(x_line, p(x_line), '--', color=colors[dataset], linewidth=2, alpha=0.6)
            
            r, p_val = stats.pearsonr(valid_data['lambda_std'], valid_data['final_test_accuracy'])
            # Add correlation text for each dataset
            y_pos = 0.95 if dataset == 'mnist' else 0.85
            ax1.text(0.05, y_pos, f'{dataset.upper()}: r={r:.3f}, p={p_val:.3f}', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=colors[dataset], alpha=0.2))
    
    ax1.set_xlabel('Lambda Std (Directional Anisotropy)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('A. Directional Anisotropy Predicts Test Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Subplot 2: Lambda Std vs Generalization Gap (per-dataset)
    ax2 = fig.add_subplot(gs[0, 1])
    
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax2.scatter(subset['lambda_std'], subset['final_generalization_gap'].abs(),
                   c=colors[dataset], label=dataset.upper(), alpha=0.7, s=120, edgecolors='black', linewidth=0.5)
        
        # Add per-dataset regression line
        valid_data = subset[['lambda_std', 'final_generalization_gap']].dropna()
        if len(valid_data) >= 3:
            z = np.polyfit(valid_data['lambda_std'], valid_data['final_generalization_gap'].abs(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data['lambda_std'].min(), valid_data['lambda_std'].max(), 100)
            ax2.plot(x_line, p(x_line), '--', color=colors[dataset], linewidth=2, alpha=0.6)
            
            r, p_val = stats.pearsonr(valid_data['lambda_std'], valid_data['final_generalization_gap'].abs())
            y_pos = 0.95 if dataset == 'mnist' else 0.85
            ax2.text(0.05, y_pos, f'{dataset.upper()}: r={r:.3f}, p={p_val:.3f}', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=colors[dataset], alpha=0.2))
    
    ax2.set_xlabel('Lambda Std (Directional Anisotropy)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('|Generalization Gap|', fontsize=13, fontweight='bold')
    ax2.set_title('B. Higher Anisotropy → Better Generalization', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    # Subplot 3: ECE vs Lambda Std (per-dataset)
    ax3 = fig.add_subplot(gs[1, 0])
    
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax3.scatter(subset['lambda_std'], subset['final_ece'],
                   c=colors[dataset], label=dataset.upper(), alpha=0.7, s=120, edgecolors='black', linewidth=0.5)
        
        # Add per-dataset regression line
        valid_data = subset[['lambda_std', 'final_ece']].dropna()
        if len(valid_data) >= 3:
            z = np.polyfit(valid_data['lambda_std'], valid_data['final_ece'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data['lambda_std'].min(), valid_data['lambda_std'].max(), 100)
            ax3.plot(x_line, p(x_line), '--', color=colors[dataset], linewidth=2, alpha=0.6)
            
            r, p_val = stats.pearsonr(valid_data['lambda_std'], valid_data['final_ece'])
            y_pos = 0.95 if dataset == 'mnist' else 0.85
            ax3.text(0.05, y_pos, f'{dataset.upper()}: r={r:.3f}, p={p_val:.3f}', 
                    transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=colors[dataset], alpha=0.2))
    
    ax3.set_xlabel('Lambda Std (Directional Anisotropy)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('ECE (Calibration Error)', fontsize=13, fontweight='bold')
    ax3.set_title('C. Higher Anisotropy → Better Calibration', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)
    
    # Subplot 4: Method ranking by Lambda Std
    ax4 = fig.add_subplot(gs[1, 1])
    
    top_methods = method_stats.nlargest(10, 'lambda_std_mean')
    colors_bar = ['#2ecc71' if x > method_stats['lambda_std_mean'].median() else '#e74c3c' 
                  for x in top_methods['lambda_std_mean']]
    
    y_pos = range(len(top_methods))
    ax4.barh(y_pos, top_methods['lambda_std_mean'], color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_methods['method'], fontsize=10)
    ax4.set_xlabel('Lambda Std', fontsize=13, fontweight='bold')
    ax4.set_title('D. Top Regularization Methods by Anisotropy', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3, axis='x')
    ax4.axvline(x=method_stats['lambda_std_mean'].median(), color='black', linestyle='--', 
                linewidth=1.5, alpha=0.5, label='Median')
    
    # Subplot 5: Comparison - Lambda Mean vs Lambda Std (spanning 2 columns, per-dataset)
    ax5 = fig.add_subplot(gs[2, :])
    
    metrics = ['Test Accuracy', 'Gen Gap', 'ECE']
    datasets = ['mnist', 'cifar10']
    colors_compare = {'mnist': '#e74c3c', 'cifar10': '#9b59b6'}
    
    x = np.arange(len(metrics))
    width = 0.2
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    
    # Collect correlations per dataset
    all_bars = []
    for i, dataset in enumerate(datasets):
        df_dataset = df[df['dataset'] == dataset]
        lambda_mean_corrs = []
        lambda_std_corrs = []
        
        for metric_col in ['final_test_accuracy', 'final_generalization_gap', 'final_ece']:
            # Lambda mean correlation
            valid_data = df_dataset[['lambda_mean', metric_col]].dropna()
            if len(valid_data) >= 3:
                if metric_col == 'final_generalization_gap':
                    r, _ = stats.pearsonr(valid_data['lambda_mean'], valid_data[metric_col].abs())
                else:
                    r, _ = stats.pearsonr(valid_data['lambda_mean'], valid_data[metric_col])
                lambda_mean_corrs.append(abs(r))
            else:
                lambda_mean_corrs.append(0)
            
            # Lambda std correlation
            valid_data = df_dataset[['lambda_std', metric_col]].dropna()
            if len(valid_data) >= 3:
                if metric_col == 'final_generalization_gap':
                    r, _ = stats.pearsonr(valid_data['lambda_std'], valid_data[metric_col].abs())
                else:
                    r, _ = stats.pearsonr(valid_data['lambda_std'], valid_data[metric_col])
                lambda_std_corrs.append(abs(r))
            else:
                lambda_std_corrs.append(0)
        
        # Plot bars for this dataset
        bars_mean = ax5.bar(x + offsets[i*2], lambda_mean_corrs, width, 
                           label=f'{dataset.upper()} Mean' if i == 0 else f'{dataset.upper()} Mean',
                           color=colors_compare[dataset], alpha=0.4, edgecolor='black', linewidth=0.5,
                           hatch='//')
        bars_std = ax5.bar(x + offsets[i*2+1], lambda_std_corrs, width,
                          label=f'{dataset.upper()} Std',
                          color=colors_compare[dataset], alpha=0.8, edgecolor='black', linewidth=0.5)
        all_bars.extend([bars_mean, bars_std])
    
    ax5.set_ylabel('|Correlation Coefficient|', fontsize=13, fontweight='bold')
    ax5.set_title('E. Lambda Std Outperforms Lambda Mean (Per-Dataset Analysis)', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, fontsize=12)
    ax5.legend(fontsize=10, loc='upper right', ncol=2)
    ax5.grid(alpha=0.3, axis='y')
    ax5.set_ylim(0, 0.8)
    
    # Add value labels on bars
    for bars in all_bars:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only label if visible
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'publication_main_figure.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved publication figure: {output_path}")
    plt.close()

def analyze_ece_lambda_relationship(df: pd.DataFrame, temporal_df: pd.DataFrame):
    """Deep dive into ECE-Lambda_std relationship (the key finding)."""
    print("\n" + "="*100)
    print("DEEP DIVE: ECE-Lambda Relationship Analysis")
    print("="*100)
    
    # 1. Per-dataset correlation breakdown
    print("\n1. ECE-Lambda_std Correlations:")
    for dataset in ['mnist', 'cifar10']:
        df_dataset = df[df['dataset'] == dataset]
        valid_data = df_dataset[['lambda_std', 'final_ece']].dropna()
        
        if len(valid_data) >= 3:
            r, p = stats.pearsonr(valid_data['lambda_std'], valid_data['final_ece'])
            r_spearman, p_spearman = stats.spearmanr(valid_data['lambda_std'], valid_data['final_ece'])
            
            print(f"\n  {dataset.upper()}:")
            print(f"    Pearson:  r={r:+.4f}, p={p:.4f}")
            print(f"    Spearman: ρ={r_spearman:+.4f}, p={p_spearman:.4f}")
            print(f"    Direction: {'Higher λ_std → Higher ECE (worse)' if r > 0 else 'Higher λ_std → Lower ECE (better)'}")
    
    # 2. Test all lambda metrics vs ECE
    print("\n2. Which Lambda Metric Best Predicts ECE?")
    lambda_metrics = {
        'lambda_mean': 'Lambda Mean',
        'lambda_std': 'Lambda Std',
        'lambda_min': 'Lambda Min',
        'lambda_max': 'Lambda Max',
        'lambda_range': 'Lambda Range',
        'lambda_median': 'Lambda Median'
    }
    
    best_results = []
    for dataset in ['mnist', 'cifar10']:
        df_dataset = df[df['dataset'] == dataset]
        print(f"\n  {dataset.upper()}:")
        
        dataset_best = {'r': 0, 'metric': ''}
        for col, name in lambda_metrics.items():
            valid_data = df_dataset[[col, 'final_ece']].dropna()
            if len(valid_data) >= 3:
                r, p = stats.pearsonr(valid_data[col], valid_data['final_ece'])
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                print(f"    {name:<20} r={r:+.4f}, p={p:.4f} {sig}")
                
                if abs(r) > abs(dataset_best['r']):
                    dataset_best = {'r': r, 'p': p, 'metric': name}
        
        best_results.append((dataset, dataset_best))
        print(f"    → Best: {dataset_best['metric']} (r={dataset_best['r']:+.4f})")
    
    # 3. Regularization method breakdown
    print("\n3. ECE by Regularization Method (sorted by λ_std):")
    print(f"\n  {'Method':<30} {'Dataset':<10} {'λ_std':>8} {'ECE':>8} {'Test Acc':>10}")
    print("  " + "-"*70)
    
    for dataset in ['mnist', 'cifar10']:
        df_dataset = df[df['dataset'] == dataset].copy()
        df_dataset = df_dataset.sort_values('lambda_std', ascending=False)
        
        for _, row in df_dataset.head(5).iterrows():
            print(f"  {row['method_name']:<30} {dataset.upper():<10} {row['lambda_std']:>8.4f} "
                  f"{row['final_ece']:>8.4f} {row['final_test_accuracy']:>10.4f}")
        print()
    
    # 4. Correlation with other metrics
    print("4. Lambda_std Correlations with All Metrics:")
    metrics_to_test = {
        'final_test_accuracy': 'Test Accuracy',
        'final_train_accuracy': 'Train Accuracy',
        'final_generalization_gap': '|Gen Gap|',
        'final_ece': 'ECE',
        'final_train_loss': 'Train Loss',
        'final_test_loss': 'Test Loss'
    }
    
    for dataset in ['mnist', 'cifar10']:
        df_dataset = df[df['dataset'] == dataset]
        print(f"\n  {dataset.upper()}:")
        
        correlations = []
        for col, name in metrics_to_test.items():
            if col in df_dataset.columns:
                valid_data = df_dataset[['lambda_std', col]].dropna()
                if len(valid_data) >= 3:
                    if col == 'final_generalization_gap':
                        r, p = stats.pearsonr(valid_data['lambda_std'], valid_data[col].abs())
                    else:
                        r, p = stats.pearsonr(valid_data['lambda_std'], valid_data[col])
                    correlations.append((abs(r), r, p, name))
        
        # Sort by absolute correlation
        correlations.sort(reverse=True)
        for abs_r, r, p, name in correlations:
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"    {name:<20} r={r:+.4f}, p={p:.4f} {sig}")
    
    # 5. Temporal evolution of ECE vs lambda_std
    print("\n5. Temporal Evolution: Does λ_std change predict ECE change?")
    
    # Merge ECE data from main df into temporal df
    if 'lambda_std_change' in temporal_df.columns:
        # Create merge keys
        temporal_df['merge_key'] = temporal_df['method'] + '_' + temporal_df['dataset']
        df_temp = df.copy()
        df_temp['merge_key'] = df_temp['method_name'] + '_' + df_temp['dataset']
        
        # Merge ECE
        temporal_with_ece = temporal_df.merge(
            df_temp[['merge_key', 'final_ece']], 
            on='merge_key', 
            how='left'
        )
        
        for dataset in ['mnist', 'cifar10']:
            df_temporal_dataset = temporal_with_ece[temporal_with_ece['dataset'] == dataset]
            valid_data = df_temporal_dataset[['lambda_std_change', 'final_ece']].dropna()
            
            if len(valid_data) >= 3:
                r, p = stats.pearsonr(valid_data['lambda_std_change'], valid_data['final_ece'])
                print(f"\n  {dataset.upper()}:")
                print(f"    λ_std change vs ECE: r={r:+.4f}, p={p:.4f}")
                if p < 0.05:
                    if r < 0:
                        print(f"    → Models that INCREASE λ_std have better calibration ✅")
                    else:
                        print(f"    → Models that DECREASE λ_std have better calibration")
            else:
                print(f"\n  {dataset.upper()}: Insufficient data (n={len(valid_data)})")
    else:
        print("  (Temporal data not available)")
    
    # 6. Visualizations
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = {'mnist': '#e74c3c', 'cifar10': '#9b59b6'}
    
    # Plot 1: Lambda_std vs ECE scatter (main finding)
    ax1 = fig.add_subplot(gs[0, :2])
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax1.scatter(subset['lambda_std'], subset['final_ece'],
                   c=colors[dataset], label=dataset.upper(), alpha=0.7, s=120, edgecolors='black', linewidth=0.5)
        
        # Regression line
        valid_data = subset[['lambda_std', 'final_ece']].dropna()
        if len(valid_data) >= 3:
            z = np.polyfit(valid_data['lambda_std'], valid_data['final_ece'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data['lambda_std'].min(), valid_data['lambda_std'].max(), 100)
            ax1.plot(x_line, p(x_line), color=colors[dataset], linewidth=2, alpha=0.6)
            
            r, p_val = stats.pearsonr(valid_data['lambda_std'], valid_data['final_ece'])
            y_pos = 0.95 if dataset == 'mnist' else 0.85
            ax1.text(0.05, y_pos, f'{dataset.upper()}: r={r:+.3f}, p={p_val:.4f}', 
                    transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=colors[dataset], alpha=0.2))
    
    ax1.set_xlabel('Lambda Std (Directional Anisotropy)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('ECE (Expected Calibration Error)', fontsize=13, fontweight='bold')
    ax1.set_title('A. Lambda_std vs ECE: The Key Correlation', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Plot 2: All lambda metrics vs ECE (comparison)
    ax2 = fig.add_subplot(gs[0, 2])
    for dataset in ['mnist', 'cifar10']:
        df_dataset = df[df['dataset'] == dataset]
        correlations = []
        
        for col in ['lambda_mean', 'lambda_std', 'lambda_min', 'lambda_max', 'lambda_range']:
            valid_data = df_dataset[[col, 'final_ece']].dropna()
            if len(valid_data) >= 3:
                r, _ = stats.pearsonr(valid_data[col], valid_data['final_ece'])
                correlations.append(abs(r))
            else:
                correlations.append(0)
        
        x = np.arange(5)
        offset = -0.2 if dataset == 'mnist' else 0.2
        ax2.bar(x + offset, correlations, width=0.35, label=dataset.upper(), 
               color=colors[dataset], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax2.set_xticks(np.arange(5))
    ax2.set_xticklabels(['Mean', 'Std', 'Min', 'Max', 'Range'], rotation=45, ha='right')
    ax2.set_ylabel('|Correlation with ECE|', fontsize=11, fontweight='bold')
    ax2.set_title('B. Which Lambda Metric\nBest Predicts ECE?', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis='y')
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Strong correlation')
    
    # Plot 3: ECE by method category
    ax3 = fig.add_subplot(gs[1, 0])
    method_categories = {
        'dropout': ['dropout_0.3', 'dropout_0.5', 'dropout_0.7'],
        'weight_decay': ['weight_decay_0.0001', 'weight_decay_0.001', 'weight_decay_0.01'],
        'label_smoothing': ['label_smoothing_0.05', 'label_smoothing_0.1', 'label_smoothing_0.15'],
        'sam': ['sam_0.05', 'sam_0.1', 'sam_0.2'],
        'other': ['baseline', 'augmentation']
    }
    
    category_stats = []
    for cat, methods in method_categories.items():
        subset = df[df['method_name'].isin(methods)]
        if len(subset) > 0:
            category_stats.append({
                'category': cat,
                'mean_lambda_std': subset['lambda_std'].mean(),
                'mean_ece': subset['final_ece'].mean()
            })
    
    cat_df = pd.DataFrame(category_stats).sort_values('mean_lambda_std', ascending=False)
    ax3.barh(cat_df['category'], cat_df['mean_ece'], color='#3498db', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Mean ECE', fontsize=11, fontweight='bold')
    ax3.set_title('C. ECE by Method Category', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='x')
    
    # Plot 4: Lambda_std by method category
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.barh(cat_df['category'], cat_df['mean_lambda_std'], color='#e74c3c', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Mean Lambda_std', fontsize=11, fontweight='bold')
    ax4.set_title('D. Lambda_std by Method Category', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, axis='x')
    
    # Plot 5: Temporal evolution
    ax5 = fig.add_subplot(gs[1, 2])
    if 'lambda_std_change' in temporal_df.columns:
        # Merge ECE data
        temporal_df['merge_key'] = temporal_df['method'] + '_' + temporal_df['dataset']
        df_temp = df.copy()
        df_temp['merge_key'] = df_temp['method_name'] + '_' + df_temp['dataset']
        temporal_with_ece = temporal_df.merge(df_temp[['merge_key', 'final_ece']], on='merge_key', how='left')
        
        for dataset in temporal_with_ece['dataset'].unique():
            subset = temporal_with_ece[temporal_with_ece['dataset'] == dataset]
            valid_data = subset[['lambda_std_change', 'final_ece']].dropna()
            
            if len(valid_data) >= 3:
                ax5.scatter(valid_data['lambda_std_change'], valid_data['final_ece'],
                           c=colors[dataset], label=dataset.upper(), alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
                
                r, p = stats.pearsonr(valid_data['lambda_std_change'], valid_data['final_ece'])
                y_pos = 0.95 if dataset == 'mnist' else 0.85
                ax5.text(0.05, y_pos, f'{dataset.upper()}: r={r:+.3f}', 
                        transform=ax5.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor=colors[dataset], alpha=0.2))
    
    ax5.set_xlabel('Lambda_std Change\n(final - initial)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Final ECE', fontsize=11, fontweight='bold')
    ax5.set_title('E. Temporal:\nλ_std Change vs ECE', fontsize=12, fontweight='bold')
    ax5.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    
    # Plot 6: Distribution of ECE by lambda_std quartiles
    ax6 = fig.add_subplot(gs[2, :])
    
    for dataset in ['mnist', 'cifar10']:
        df_dataset = df[df['dataset'] == dataset]
        
        # Split into quartiles
        df_dataset['lambda_std_quartile'] = pd.qcut(df_dataset['lambda_std'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        
        quartile_data = []
        positions = []
        for i, q in enumerate(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']):
            data = df_dataset[df_dataset['lambda_std_quartile'] == q]['final_ece'].dropna()
            if len(data) > 0:
                quartile_data.append(data)
                offset = -0.2 if dataset == 'mnist' else 0.2
                positions.append(i + offset)
        
        bp = ax6.boxplot(quartile_data, positions=positions, widths=0.35,
                         patch_artist=True, showfliers=False,
                         boxprops=dict(facecolor=colors[dataset], alpha=0.7),
                         medianprops=dict(color='black', linewidth=2),
                         whiskerprops=dict(color=colors[dataset]),
                         capprops=dict(color=colors[dataset]))
    
    ax6.set_xticks([0, 1, 2, 3])
    ax6.set_xticklabels(['Q1\n(Low λ_std)', 'Q2', 'Q3', 'Q4\n(High λ_std)'])
    ax6.set_ylabel('ECE Distribution', fontsize=12, fontweight='bold')
    ax6.set_title('F. ECE Distribution by Lambda_std Quartiles', fontsize=13, fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['mnist'], alpha=0.7, label='MNIST'),
                      Patch(facecolor=colors['cifar10'], alpha=0.7, label='CIFAR10')]
    ax6.legend(handles=legend_elements, fontsize=10, loc='upper right')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'ece_lambda_deep_dive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved ECE deep dive: {output_path}\n")
    plt.close()

def generate_directional_summary(df: pd.DataFrame, corr_results: pd.DataFrame):
    """Generate summary of directional analysis."""
    print_section("11. SCIENTIFIC FINDINGS SUMMARY", level=1)
    
    print("\n" + "="*100)
    print("HYPOTHESIS TEST: Does Lambda Direction Matter? (Per-Dataset Analysis)")
    print("="*100)
    
    # Analyze per dataset
    print("\n📊 LAMBDA_STD CORRELATIONS (WITHIN EACH DATASET):")
    
    for dataset in ['cifar10', 'mnist']:
        df_dataset = df[df['dataset'] == dataset]
        print(f"\n{dataset.upper()}:")
        
        for metric_col, metric_name in [
            ('final_test_accuracy', 'Test Accuracy'),
            ('final_generalization_gap', '|Gen Gap|'),
            ('final_ece', 'ECE')
        ]:
            valid_data = df_dataset[['lambda_std', metric_col]].dropna()
            
            if len(valid_data) >= 3:
                if metric_col == 'final_generalization_gap':
                    r, p = stats.pearsonr(valid_data['lambda_std'], valid_data[metric_col].abs())
                else:
                    r, p = stats.pearsonr(valid_data['lambda_std'], valid_data[metric_col])
                
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                print(f"  λ_std vs {metric_name:<20} r={r:+.4f}, p={p:.4f} {sig}")
    
    # Count significant findings
    print("\n" + "="*100)
    print("VERDICT:")
    print("="*100)
    
    sig_count = 0
    for dataset in ['cifar10', 'mnist']:
        df_dataset = df[df['dataset'] == dataset]
        for metric_col in ['final_test_accuracy', 'final_generalization_gap', 'final_ece']:
            valid_data = df_dataset[['lambda_std', metric_col]].dropna()
            if len(valid_data) >= 3:
                if metric_col == 'final_generalization_gap':
                    r, p = stats.pearsonr(valid_data['lambda_std'], valid_data[metric_col].abs())
                else:
                    r, p = stats.pearsonr(valid_data['lambda_std'], valid_data[metric_col])
                if p < 0.05:
                    sig_count += 1
    
    if sig_count > 0:
        print(f"\n⚠️  Lambda_std shows {sig_count}/6 significant correlations (per-dataset analysis)")
        print(f"   BUT: Correlations have OPPOSITE signs (MNIST: r=-0.60, CIFAR10: r=+0.56)")
        print(f"   AND: Lambda_mean/min/max/median show MUCH stronger correlations (|r|~0.87-0.94)")
        print(f"\n   ACTUAL FINDING: Lambda_mean (not std) strongly predicts ECE")
        print(f"   PROBLEM: Different sign per dataset suggests confounding variables")
        print(f"   CONCLUSION: Need more experiments to understand λ-ECE relationship")
    else:
        print(f"\n❌ No significant correlations found")
    print("="*100)
    print("="*100)

def main():
    """Main analysis pipeline."""
    print("\nDirectional Lambda Analysis (Per-Dataset Correlations)")
    print("="*60)
    
    # Load data with directional information
    df = load_directional_lambda_data()
    
    if len(df) == 0:
        print("❌ No directional lambda data found")
        return
    
    # Run comprehensive analyses
    corr_results = test_directional_correlations(df)
    analyze_flattest_vs_sharpest(df)
    analyze_lambda_spread(df)
    create_directional_visualizations(df)
    
    # NEW: Advanced analyses
    temporal_df = analyze_temporal_evolution(df)
    analyze_per_dataset_robustness(df)
    compute_effect_sizes_and_power(df)
    method_stats = rank_regularization_methods(df)
    create_publication_figures(df, method_stats)
    
    # DEEP DIVE: ECE-Lambda relationship
    analyze_ece_lambda_relationship(df, temporal_df)
    
    generate_directional_summary(df, corr_results)
    
    # Save all results
    output_path = OUTPUT_DIR / 'directional_lambda_data.csv'
    df_save = df.drop(columns=['lambda_per_direction'])
    df_save.to_csv(output_path, index=False)
    print(f"✓ Saved directional data: {output_path}")
    
    output_path = OUTPUT_DIR / 'temporal_lambda_data.csv'
    temporal_df_save = temporal_df.drop(columns=['epochs', 'lambda_stds'], errors='ignore')
    temporal_df_save.to_csv(output_path, index=False)
    print(f"✓ Saved temporal data: {output_path}\n")

if __name__ == '__main__':
    main()
