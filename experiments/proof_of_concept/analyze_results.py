"""
Comprehensive analysis script for Proof of Concept experiment results.

Analyzes the relationship between lambda (loss landscape curvature) and 
model performance metrics (accuracy, generalization, calibration).

Usage:
    python analyze_results.py --dataset cifar10
    python analyze_results.py --dataset mnist
    python analyze_results.py --dataset both
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_results(results_dir: Path, dataset: str = None):
    """
    Load all experiment results from summary.json files.
    
    Args:
        results_dir: Path to results directory
        dataset: Optional dataset filter ('cifar10', 'mnist', or None for both)
    
    Returns:
        DataFrame with all results
    """
    all_results = []
    
    # Find all dataset directories
    dataset_dirs = []
    if dataset:
        dataset_path = results_dir / dataset
        if dataset_path.exists():
            dataset_dirs.append((dataset, dataset_path))
    else:
        for ds in ['cifar10', 'mnist']:
            dataset_path = results_dir / ds
            if dataset_path.exists():
                dataset_dirs.append((ds, dataset_path))
    
    # Load results from each method
    for ds_name, ds_path in dataset_dirs:
        for method_dir in ds_path.iterdir():
            if not method_dir.is_dir():
                continue
            
            summary_file = method_dir / 'summary.json'
            if summary_file.exists():
                import json
                with open(summary_file, 'r') as f:
                    result = json.load(f)
                    all_results.append(result)
    
    if not all_results:
        raise ValueError(f"No results found in {results_dir}")
    
    return pd.DataFrame(all_results)


def compute_correlations(df, dataset_name):
    """
    Compute correlations between lambda metrics and performance metrics.
    
    Args:
        df: DataFrame with results
        dataset_name: Name of dataset for display
    
    Returns:
        DataFrame with correlation results
    """
    lambda_metrics = {
        'final_lambda_mean': 'Lambda Mean',
        'final_lambda_std': 'Lambda Std'
    }
    
    performance_metrics = {
        'final_test_accuracy': 'Test Accuracy',
        'final_generalization_gap': 'Generalization Gap',
        'final_ece': 'ECE (Calibration)',
        'final_train_accuracy': 'Train Accuracy'
    }
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} CORRELATIONS")
    print(f"{'='*80}")
    
    for perf_col, perf_name in performance_metrics.items():
        print(f"\n{perf_name}:")
        print("-" * 60)
        
        for lambda_col, lambda_name in lambda_metrics.items():
            # Remove any NaN values
            mask = ~(df[perf_col].isna() | df[lambda_col].isna())
            x = df.loc[mask, lambda_col]
            y = df.loc[mask, perf_col]
            
            if len(x) < 3:
                print(f"  {lambda_name}: Insufficient data")
                continue
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(x, y)
            
            # Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(x, y)
            
            # Determine significance
            is_sig = pearson_p < 0.05
            sig_marker = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else ""
            
            print(f"  {lambda_name}:")
            print(f"    Pearson:  r = {pearson_r:7.4f}, p = {pearson_p:.4f} {sig_marker}")
            print(f"    Spearman: r = {spearman_r:7.4f}, p = {spearman_p:.4f}")
            
            results.append({
                'Dataset': dataset_name,
                'Performance_Metric': perf_name,
                'Lambda_Metric': lambda_name,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p,
                'Significant': is_sig,
                'N': len(x)
            })
    
    return pd.DataFrame(results)


def create_correlation_heatmap(df, dataset_name, output_path):
    """Create correlation heatmap for lambda vs performance metrics."""
    
    lambda_metrics = ['final_lambda_mean', 'final_lambda_std']
    performance_metrics = {
        'final_test_accuracy': 'Test Accuracy',
        'final_generalization_gap': 'Generalization Gap',
        'final_ece': 'ECE',
        'final_train_accuracy': 'Train Accuracy'
    }
    
    # Create correlation matrix
    corr_data = []
    for perf_col, perf_name in performance_metrics.items():
        row = []
        for lambda_col in lambda_metrics:
            mask = ~(df[perf_col].isna() | df[lambda_col].isna())
            x = df.loc[mask, lambda_col]
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
        columns=['Lambda Mean', 'Lambda Std']
    )
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title(f'{dataset_name.upper()} - Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved heatmap: {output_path}")


def create_scatter_plots(df, dataset_name, output_path):
    """Create scatter plots for lambda mean vs each performance metric."""
    
    performance_metrics = {
        'final_test_accuracy': 'Test Accuracy',
        'final_generalization_gap': 'Generalization Gap',
        'final_ece': 'ECE (Calibration)',
        'final_train_accuracy': 'Train Accuracy'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (perf_col, perf_name) in enumerate(performance_metrics.items()):
        ax = axes[idx]
        
        # Plot scatter
        ax.scatter(df['final_lambda_mean'], df[perf_col], alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        # Compute correlation
        mask = ~(df['final_lambda_mean'].isna() | df[perf_col].isna())
        x = df.loc[mask, 'final_lambda_mean']
        y = df.loc[mask, perf_col]
        r, p = stats.pearsonr(x, y)
        
        # Add trend line
        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p_fit(x_line), 'r--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('Lambda Mean (Log Curvature Rate)', fontsize=11)
        ax.set_ylabel(perf_name, fontsize=11)
        
        # Significance marker
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f'{perf_name}\nr = {r:.3f}, p = {p:.4f} {sig}', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_name.upper()} - Lambda Mean vs Performance Metrics', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved scatter plots: {output_path}")


def create_method_comparison(df, dataset_name, output_path):
    """Create sorted bar chart showing lambda and ECE for each method."""
    
    # Sort by lambda mean
    df_sorted = df.sort_values('final_lambda_mean').reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    x_pos = np.arange(len(df_sorted))
    
    # Plot 1: Lambda Mean
    bars1 = ax1.barh(x_pos, df_sorted['final_lambda_mean'], color='steelblue', alpha=0.7)
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(df_sorted['method_name'], fontsize=9)
    ax1.set_xlabel('Lambda Mean (Log Curvature Rate)', fontsize=11)
    ax1.set_title('Flatness: Flatter (more negative) → Sharper', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 2: ECE (same order as lambda)
    colors = ['green' if ece < 0.15 else 'orange' if ece < 0.30 else 'red' 
              for ece in df_sorted['final_ece']]
    bars2 = ax2.barh(x_pos, df_sorted['final_ece'], color=colors, alpha=0.7)
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels([''] * len(df_sorted))  # No labels on right plot
    ax2.set_xlabel('ECE (Expected Calibration Error)', fontsize=11)
    ax2.set_title('Calibration: Lower is Better', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'{dataset_name.upper()} - Methods Ranked by Flatness (Lambda Mean)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved method comparison: {output_path}")


def print_summary_statistics(df, dataset_name):
    """Print summary statistics for the dataset."""
    
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"\nNumber of methods: {len(df)}")
    
    print(f"\nTest Accuracy:")
    print(f"  Range: [{df['final_test_accuracy'].min():.4f}, {df['final_test_accuracy'].max():.4f}]")
    print(f"  Mean:  {df['final_test_accuracy'].mean():.4f} ± {df['final_test_accuracy'].std():.4f}")
    
    print(f"\nGeneralization Gap (Train - Test):")
    print(f"  Range: [{df['final_generalization_gap'].min():.4f}, {df['final_generalization_gap'].max():.4f}]")
    print(f"  Mean:  {df['final_generalization_gap'].mean():.4f} ± {df['final_generalization_gap'].std():.4f}")
    
    print(f"\nECE (Calibration Error):")
    print(f"  Range: [{df['final_ece'].min():.4f}, {df['final_ece'].max():.4f}]")
    print(f"  Mean:  {df['final_ece'].mean():.4f} ± {df['final_ece'].std():.4f}")
    
    print(f"\nLambda Mean (Log Curvature):")
    print(f"  Range: [{df['final_lambda_mean'].min():.4f}, {df['final_lambda_mean'].max():.4f}]")
    print(f"  Mean:  {df['final_lambda_mean'].mean():.4f} ± {df['final_lambda_mean'].std():.4f}")
    
    print(f"\nLambda Std (Curvature Spread):")
    print(f"  Range: [{df['final_lambda_std'].min():.4f}, {df['final_lambda_std'].max():.4f}]")
    print(f"  Mean:  {df['final_lambda_std'].mean():.4f} ± {df['final_lambda_std'].std():.4f}")


def print_top_methods(df, dataset_name):
    """Print top methods by different metrics."""
    
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} TOP METHODS")
    print(f"{'='*80}")
    
    print(f"\nTop 5 by Test Accuracy:")
    print("-" * 60)
    top_acc = df.nlargest(5, 'final_test_accuracy')[['method_name', 'final_test_accuracy', 'final_lambda_mean', 'final_ece']]
    for idx, row in top_acc.iterrows():
        print(f"  {row['method_name']:25s}  Acc: {row['final_test_accuracy']:.4f}  λ: {row['final_lambda_mean']:7.4f}  ECE: {row['final_ece']:.4f}")
    
    print(f"\nTop 5 by Calibration (Lowest ECE):")
    print("-" * 60)
    top_cal = df.nsmallest(5, 'final_ece')[['method_name', 'final_ece', 'final_lambda_mean', 'final_test_accuracy']]
    for idx, row in top_cal.iterrows():
        print(f"  {row['method_name']:25s}  ECE: {row['final_ece']:.4f}  λ: {row['final_lambda_mean']:7.4f}  Acc: {row['final_test_accuracy']:.4f}")
    
    print(f"\nTop 5 by Flatness (Most Negative Lambda):")
    print("-" * 60)
    top_flat = df.nsmallest(5, 'final_lambda_mean')[['method_name', 'final_lambda_mean', 'final_ece', 'final_test_accuracy']]
    for idx, row in top_flat.iterrows():
        print(f"  {row['method_name']:25s}  λ: {row['final_lambda_mean']:7.4f}  ECE: {row['final_ece']:.4f}  Acc: {row['final_test_accuracy']:.4f}")


def main():
    """Main analysis routine."""
    parser = argparse.ArgumentParser(
        description='Analyze Proof of Concept experiment results'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cifar10', 'mnist', 'both'],
        default='both',
        help='Which dataset to analyze (default: both)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/proof_of_concept',
        help='Directory containing experiment results'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PROOF OF CONCEPT RESULTS ANALYSIS")
    print("="*80)
    print(f"\nResults directory: {results_dir}")
    print(f"Analysis output:   {analysis_dir}")
    
    # Determine datasets to analyze
    datasets_to_analyze = []
    if args.dataset == 'both':
        for ds in ['cifar10', 'mnist']:
            if (results_dir / ds).exists():
                datasets_to_analyze.append(ds)
    else:
        if (results_dir / args.dataset).exists():
            datasets_to_analyze.append(args.dataset)
    
    if not datasets_to_analyze:
        print(f"\nERROR: No results found for dataset(s): {args.dataset}")
        return
    
    # Load all results
    all_results_dfs = {}
    all_correlations = []
    
    for dataset in datasets_to_analyze:
        print(f"\n{'='*80}")
        print(f"ANALYZING {dataset.upper()}")
        print(f"{'='*80}")
        
        # Load results
        df = load_results(results_dir, dataset=dataset)
        all_results_dfs[dataset] = df
        
        print(f"\nLoaded {len(df)} experiments")
        
        # Print summary statistics
        print_summary_statistics(df, dataset)
        
        # Compute correlations
        corr_df = compute_correlations(df, dataset)
        all_correlations.append(corr_df)
        
        # Create visualizations
        print(f"\nGenerating visualizations...")
        
        heatmap_path = analysis_dir / f'{dataset}_correlation_heatmap.png'
        create_correlation_heatmap(df, dataset, heatmap_path)
        
        scatter_path = analysis_dir / f'{dataset}_scatter_plots.png'
        create_scatter_plots(df, dataset, scatter_path)
        
        comparison_path = analysis_dir / f'{dataset}_method_comparison.png'
        create_method_comparison(df, dataset, comparison_path)
        
        # Print top methods
        print_top_methods(df, dataset)
    
    # Save consolidated results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    # Save all results to CSV
    all_results_df = pd.concat([df.assign(dataset=ds) for ds, df in all_results_dfs.items()], ignore_index=True)
    results_csv_path = analysis_dir / 'all_results.csv'
    all_results_df.to_csv(results_csv_path, index=False)
    print(f"\n  Saved all results: {results_csv_path}")
    
    # Save correlations
    all_corr_df = pd.concat(all_correlations, ignore_index=True)
    corr_csv_path = analysis_dir / 'correlations.csv'
    all_corr_df.to_csv(corr_csv_path, index=False)
    print(f"  Saved correlations: {corr_csv_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {analysis_dir}")


if __name__ == '__main__':
    main()
