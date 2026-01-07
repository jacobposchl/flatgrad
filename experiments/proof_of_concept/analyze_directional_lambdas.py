"""
Analyze directional lambda statistics to see if averaging loses valuable information.

This script examines the distribution of lambda values across different random directions
to determine if higher-order statistics (beyond mean/std) correlate with performance.

Usage:
    python analyze_directional_lambdas.py --dataset cifar10
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_directional_lambda_data(results_dir: Path, dataset: str):
    """
    Load per-direction lambda values from all experiments.
    
    Args:
        results_dir: Path to results directory
        dataset: Dataset name ('cifar10' or 'mnist')
    
    Returns:
        DataFrame with directional statistics
    """
    dataset_dir = results_dir / dataset
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    
    all_data = []
    
    for method_dir in dataset_dir.iterdir():
        if not method_dir.is_dir():
            continue
        
        # Load lambda data
        lambda_file = method_dir / 'lambda_data.npz'
        summary_file = method_dir / 'summary.json'
        
        if not lambda_file.exists() or not summary_file.exists():
            continue
        
        # Load summary (for performance metrics)
        import json
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Load lambda data
        data = np.load(lambda_file, allow_pickle=True)
        
        # Get final epoch lambda values per direction
        lambda_values_per_epoch = data.get('lambda_values_per_epoch', None)
        
        if lambda_values_per_epoch is None or len(lambda_values_per_epoch) == 0:
            print(f"  Warning: No directional data for {method_dir.name}")
            continue
        
        # Get lambda values from final measurement
        final_lambda_values = lambda_values_per_epoch[-1]
        
        if len(final_lambda_values) == 0:
            print(f"  Warning: Empty lambda values for {method_dir.name}")
            continue
        
        # Compute directional statistics - ensure it's a numeric array
        lambda_array = np.array(final_lambda_values, dtype=np.float64)
        
        directional_stats = {
            'method_name': summary['method_name'],
            'dataset': dataset,
            
            # Performance metrics
            'test_accuracy': summary['final_test_accuracy'],
            'generalization_gap': summary['final_generalization_gap'],
            'ece': summary['final_ece'],
            'train_accuracy': summary['final_train_accuracy'],
            
            # Basic statistics (from summary)
            'lambda_mean': summary['final_lambda_mean'],
            'lambda_std': summary['final_lambda_std'],
            
            # Directional statistics
            'n_directions': len(lambda_array),
            'lambda_min': np.min(lambda_array),
            'lambda_max': np.max(lambda_array),
            'lambda_range': np.max(lambda_array) - np.min(lambda_array),
            'lambda_median': np.median(lambda_array),
            'lambda_q25': np.percentile(lambda_array, 25),
            'lambda_q75': np.percentile(lambda_array, 75),
            'lambda_iqr': np.percentile(lambda_array, 75) - np.percentile(lambda_array, 25),
            
            # Higher-order moments
            'lambda_skewness': stats.skew(lambda_array),
            'lambda_kurtosis': stats.kurtosis(lambda_array),
            
            # Variation measures
            'lambda_cv': np.std(lambda_array) / abs(np.mean(lambda_array)) if np.mean(lambda_array) != 0 else np.nan,
            'lambda_mad': np.median(np.abs(lambda_array - np.median(lambda_array))),  # Median absolute deviation
            
            # Extreme values
            'lambda_min_max_ratio': np.min(lambda_array) / np.max(lambda_array) if np.max(lambda_array) != 0 else np.nan,
            
            # Full distribution for later analysis
            'lambda_values': lambda_array
        }
        
        all_data.append(directional_stats)
    
    return pd.DataFrame(all_data)


def compute_extended_correlations(df, dataset_name):
    """
    Compute correlations between all directional statistics and performance metrics.
    
    Args:
        df: DataFrame with directional statistics
        dataset_name: Name of dataset for display
    
    Returns:
        DataFrame with correlation results
    """
    # Directional statistics to test
    directional_features = {
        'lambda_mean': 'Lambda Mean',
        'lambda_std': 'Lambda Std',
        'lambda_min': 'Lambda Min (Flattest)',
        'lambda_max': 'Lambda Max (Sharpest)',
        'lambda_range': 'Lambda Range',
        'lambda_median': 'Lambda Median',
        'lambda_iqr': 'Lambda IQR',
        'lambda_skewness': 'Lambda Skewness',
        'lambda_kurtosis': 'Lambda Kurtosis',
        'lambda_cv': 'Lambda CV',
        'lambda_mad': 'Lambda MAD',
    }
    
    performance_metrics = {
        'test_accuracy': 'Test Accuracy',
        'generalization_gap': 'Generalization Gap',
        'ece': 'ECE (Calibration)',
        'train_accuracy': 'Train Accuracy'
    }
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} DIRECTIONAL CORRELATIONS")
    print(f"{'='*80}")
    
    for perf_col, perf_name in performance_metrics.items():
        print(f"\n{perf_name}:")
        print("-" * 80)
        
        # Sort by correlation strength
        correlations = []
        
        for feat_col, feat_name in directional_features.items():
            # Remove any NaN values
            mask = ~(df[perf_col].isna() | df[feat_col].isna())
            x = df.loc[mask, feat_col]
            y = df.loc[mask, perf_col]
            
            if len(x) < 3:
                continue
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(x, y)
            
            # Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(x, y)
            
            correlations.append({
                'feature': feat_name,
                'feature_col': feat_col,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'abs_r': abs(pearson_r)
            })
            
            results.append({
                'Dataset': dataset_name,
                'Performance_Metric': perf_name,
                'Directional_Feature': feat_name,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p,
                'Significant': pearson_p < 0.05,
                'N': len(x)
            })
        
        # Sort by absolute correlation and print top 5
        correlations.sort(key=lambda x: x['abs_r'], reverse=True)
        
        for i, corr in enumerate(correlations[:5], 1):
            sig = "***" if corr['pearson_p'] < 0.001 else "**" if corr['pearson_p'] < 0.01 else "*" if corr['pearson_p'] < 0.05 else ""
            print(f"  {i}. {corr['feature']:25s}  r={corr['pearson_r']:7.4f}, p={corr['pearson_p']:.4f} {sig}")
    
    return pd.DataFrame(results)


def create_directional_heatmap(df, dataset_name, output_path):
    """Create comprehensive correlation heatmap."""
    
    directional_features = [
        'lambda_mean', 'lambda_std', 'lambda_min', 'lambda_max', 
        'lambda_range', 'lambda_median', 'lambda_iqr', 
        'lambda_skewness', 'lambda_kurtosis', 'lambda_cv'
    ]
    
    feature_names = [
        'Mean', 'Std', 'Min', 'Max', 
        'Range', 'Median', 'IQR', 
        'Skewness', 'Kurtosis', 'CV'
    ]
    
    performance_metrics = {
        'test_accuracy': 'Test Acc',
        'generalization_gap': 'Gen Gap',
        'ece': 'ECE',
        'train_accuracy': 'Train Acc'
    }
    
    # Compute correlation matrix
    corr_data = []
    for perf_col, perf_name in performance_metrics.items():
        row = []
        for feat_col in directional_features:
            mask = ~(df[perf_col].isna() | df[feat_col].isna())
            x = df.loc[mask, feat_col]
            y = df.loc[mask, perf_col]
            if len(x) >= 3:
                r, _ = stats.pearsonr(x, y)
                row.append(r)
            else:
                row.append(np.nan)
        corr_data.append(row)
    
    corr_matrix = pd.DataFrame(
        corr_data,
        index=list(performance_metrics.values()),
        columns=feature_names
    )
    
    # Plot heatmap
    plt.figure(figsize=(14, 5))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation'},
                annot_kws={'size': 9})
    plt.title(f'{dataset_name.upper()} - Directional Lambda Statistics vs Performance', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Lambda Directional Statistics', fontsize=11)
    plt.ylabel('Performance Metrics', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved directional heatmap: {output_path}")


def create_distribution_plots(df, dataset_name, output_path):
    """Plot lambda distributions for different methods."""
    
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    axes = axes.flatten()
    
    # Sort by lambda mean
    df_sorted = df.sort_values('lambda_mean')
    
    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        if idx >= 18:
            break
        
        ax = axes[idx]
        lambda_values = row['lambda_values']
        
        # Plot histogram
        ax.hist(lambda_values, bins=10, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(row['lambda_mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {row["lambda_mean"]:.2f}')
        ax.axvline(row['lambda_median'], color='orange', linestyle='--', linewidth=2, label=f'Median: {row["lambda_median"]:.2f}')
        
        ax.set_title(f'{row["method_name"]}\nECE: {row["ece"]:.3f}', fontsize=9)
        ax.set_xlabel('Lambda', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)
    
    # Hide unused subplots
    for idx in range(len(df_sorted), 18):
        axes[idx].axis('off')
    
    plt.suptitle(f'{dataset_name.upper()} - Lambda Distributions Across Directions', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved distribution plots: {output_path}")


def create_key_scatter_plots(df, dataset_name, output_path):
    """Create scatter plots for most interesting directional features."""
    
    # Focus on features that might capture directional heterogeneity
    key_features = [
        ('lambda_range', 'Lambda Range (Directional Spread)'),
        ('lambda_iqr', 'Lambda IQR'),
        ('lambda_skewness', 'Lambda Skewness'),
        ('lambda_cv', 'Lambda Coefficient of Variation')
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for col_idx, (feature, feature_name) in enumerate(key_features):
        # ECE vs feature
        ax = axes[0, col_idx]
        ax.scatter(df[feature], df['ece'], alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        mask = ~(df[feature].isna() | df['ece'].isna())
        x = df.loc[mask, feature]
        y = df.loc[mask, 'ece']
        r, p = stats.pearsonr(x, y)
        
        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p_fit(x_line), 'r--', alpha=0.5, linewidth=2)
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f'ECE vs {feature_name}\nr={r:.3f}, p={p:.4f} {sig}', fontsize=10)
        ax.set_xlabel(feature_name, fontsize=9)
        ax.set_ylabel('ECE', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Generalization Gap vs feature
        ax = axes[1, col_idx]
        ax.scatter(df[feature], df['generalization_gap'], alpha=0.6, s=100, edgecolors='black', linewidth=0.5, color='orange')
        
        mask = ~(df[feature].isna() | df['generalization_gap'].isna())
        x = df.loc[mask, feature]
        y = df.loc[mask, 'generalization_gap']
        r, p = stats.pearsonr(x, y)
        
        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p_fit(x_line), 'r--', alpha=0.5, linewidth=2)
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f'Gen Gap vs {feature_name}\nr={r:.3f}, p={p:.4f} {sig}', fontsize=10)
        ax.set_xlabel(feature_name, fontsize=9)
        ax.set_ylabel('Generalization Gap', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_name.upper()} - Directional Heterogeneity Features', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved key scatter plots: {output_path}")


def main():
    """Main analysis routine."""
    parser = argparse.ArgumentParser(
        description='Analyze directional lambda statistics'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cifar10', 'mnist', 'both'],
        default='cifar10',
        help='Which dataset to analyze (default: cifar10)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/proof_of_concept',
        help='Directory containing experiment results'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("DIRECTIONAL LAMBDA ANALYSIS")
    print("="*80)
    print("\nHypothesis: Averaging lambda across directions loses valuable information")
    print("about directional heterogeneity and distribution shape.")
    
    # Determine datasets
    datasets = []
    if args.dataset == 'both':
        for ds in ['cifar10', 'mnist']:
            if (results_dir / ds).exists():
                datasets.append(ds)
    else:
        if (results_dir / args.dataset).exists():
            datasets.append(args.dataset)
    
    all_correlations = []
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"LOADING {dataset.upper()} DIRECTIONAL DATA")
        print(f"{'='*80}")
        
        # Load directional data
        df = load_directional_lambda_data(results_dir, dataset)
        print(f"\nLoaded {len(df)} experiments with directional data")
        print(f"Average {df['n_directions'].mean():.1f} directions per experiment")
        
        # Compute correlations
        corr_df = compute_extended_correlations(df, dataset)
        all_correlations.append(corr_df)
        
        # Create visualizations
        print(f"\nGenerating visualizations...")
        
        heatmap_path = analysis_dir / f'{dataset}_directional_heatmap.png'
        create_directional_heatmap(df, dataset, heatmap_path)
        
        dist_path = analysis_dir / f'{dataset}_lambda_distributions.png'
        create_distribution_plots(df, dataset, dist_path)
        
        scatter_path = analysis_dir / f'{dataset}_directional_scatter.png'
        create_key_scatter_plots(df, dataset, scatter_path)
        
        # Save directional statistics
        stats_csv = analysis_dir / f'{dataset}_directional_statistics.csv'
        df_save = df.drop(columns=['lambda_values'])  # Drop array column for CSV
        df_save.to_csv(stats_csv, index=False)
        print(f"  Saved directional statistics: {stats_csv}")
    
    # Save all correlations
    all_corr_df = pd.concat(all_correlations, ignore_index=True)
    corr_csv = analysis_dir / 'directional_correlations.csv'
    all_corr_df.to_csv(corr_csv, index=False)
    print(f"\n  Saved all correlations: {corr_csv}")
    
    print(f"\n{'='*80}")
    print("DIRECTIONAL ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
