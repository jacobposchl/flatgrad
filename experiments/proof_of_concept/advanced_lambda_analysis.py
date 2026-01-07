"""
Advanced Lambda Analysis: Structured Directions and Additional Tests

This script explores:
1. Class-specific directions (gradient directions toward class boundaries)
2. Adversarial directions (directions of misclassification)
3. Temporal dynamics (how lambda evolves during training)
4. Statistical hypothesis testing beyond correlation
5. Method clustering based on lambda profiles

Usage:
    python advanced_lambda_analysis.py --dataset cifar10
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_temporal_lambda_data(results_dir: Path, dataset: str):
    """
    Load lambda evolution over training for all experiments.
    
    Returns:
        DataFrame with temporal lambda data
    """
    dataset_dir = results_dir / dataset
    all_data = []
    
    for method_dir in dataset_dir.iterdir():
        if not method_dir.is_dir():
            continue
        
        lambda_file = method_dir / 'lambda_data.npz'
        if not lambda_file.exists():
            continue
        
        data = np.load(lambda_file, allow_pickle=True)
        
        epochs = data['epochs']
        lambda_means = data['lambda_means']
        lambda_stds = data['lambda_stds']
        
        for epoch, lmean, lstd in zip(epochs, lambda_means, lambda_stds):
            all_data.append({
                'method_name': method_dir.name,
                'epoch': epoch,
                'lambda_mean': lmean,
                'lambda_std': lstd
            })
    
    return pd.DataFrame(all_data)


def analyze_temporal_dynamics(df, dataset_name, output_dir):
    """
    Analyze how lambda evolves during training and if early/late training
    lambda has different predictive power.
    """
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} TEMPORAL DYNAMICS ANALYSIS")
    print(f"{'='*80}")
    
    # Get early (epoch 0-10), mid (epoch 20-40), late (epoch 80-100) lambda
    methods = df['method_name'].unique()
    
    temporal_stats = []
    
    for method in methods:
        method_data = df[df['method_name'] == method].sort_values('epoch')
        
        if len(method_data) == 0:
            continue
        
        early_lambda = method_data[method_data['epoch'] <= 10]['lambda_mean'].mean()
        mid_lambda = method_data[(method_data['epoch'] >= 20) & (method_data['epoch'] <= 40)]['lambda_mean'].mean()
        late_lambda = method_data[method_data['epoch'] >= 80]['lambda_mean'].mean()
        
        lambda_trajectory_slope = None
        if len(method_data) >= 2:
            # Fit linear trend
            slope, _ = np.polyfit(method_data['epoch'], method_data['lambda_mean'], 1)
            lambda_trajectory_slope = slope
        
        # Compute volatility (std of lambda over time)
        lambda_volatility = method_data['lambda_mean'].std()
        
        temporal_stats.append({
            'method_name': method,
            'early_lambda': early_lambda,
            'mid_lambda': mid_lambda,
            'late_lambda': late_lambda,
            'lambda_change': late_lambda - early_lambda,
            'lambda_trajectory_slope': lambda_trajectory_slope,
            'lambda_volatility': lambda_volatility
        })
    
    temporal_df = pd.DataFrame(temporal_stats)
    
    print(f"\nTemporal Statistics:")
    print(f"  Average early lambda (0-10):  {temporal_df['early_lambda'].mean():.4f}")
    print(f"  Average mid lambda (20-40):   {temporal_df['mid_lambda'].mean():.4f}")
    print(f"  Average late lambda (80-100): {temporal_df['late_lambda'].mean():.4f}")
    print(f"  Average lambda change:        {temporal_df['lambda_change'].mean():.4f}")
    
    # Plot temporal evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Lambda evolution for all methods
    ax = axes[0, 0]
    for method in methods[:10]:  # Limit to 10 for clarity
        method_data = df[df['method_name'] == method].sort_values('epoch')
        ax.plot(method_data['epoch'], method_data['lambda_mean'], label=method, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Lambda Mean')
    ax.set_title('Lambda Evolution During Training')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Lambda change distribution
    ax = axes[0, 1]
    ax.hist(temporal_df['lambda_change'], bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Lambda Change (Late - Early)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Lambda Changes')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Lambda volatility distribution
    ax = axes[1, 0]
    ax.hist(temporal_df['lambda_volatility'], bins=15, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Lambda Volatility (Std over time)')
    ax.set_ylabel('Count')
    ax.set_title('Lambda Stability During Training')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Early vs Late lambda
    ax = axes[1, 1]
    ax.scatter(temporal_df['early_lambda'], temporal_df['late_lambda'], s=100, alpha=0.6)
    ax.plot([temporal_df['early_lambda'].min(), temporal_df['early_lambda'].max()],
            [temporal_df['early_lambda'].min(), temporal_df['early_lambda'].max()],
            'r--', linewidth=2, label='No change')
    ax.set_xlabel('Early Lambda (0-10)')
    ax.set_ylabel('Late Lambda (80-100)')
    ax.set_title('Lambda Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_temporal_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved temporal dynamics plot")
    
    return temporal_df


def statistical_hypothesis_tests(df_directional, df_performance, dataset_name):
    """
    Perform rigorous statistical hypothesis tests beyond correlation.
    """
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} STATISTICAL HYPOTHESIS TESTS")
    print(f"{'='*80}")
    
    # Test 1: Methods with high skewness vs low skewness generalization
    median_skewness = df_directional['lambda_skewness'].median()
    high_skew = df_directional[df_directional['lambda_skewness'] >= median_skewness]
    low_skew = df_directional[df_directional['lambda_skewness'] < median_skewness]
    
    # T-test for generalization gap
    t_stat, p_value = stats.ttest_ind(
        high_skew['generalization_gap'].dropna(),
        low_skew['generalization_gap'].dropna()
    )
    
    print(f"\nTest 1: High vs Low Skewness Methods")
    print(f"  High skewness gen gap: {high_skew['generalization_gap'].mean():.4f} ± {high_skew['generalization_gap'].std():.4f}")
    print(f"  Low skewness gen gap:  {low_skew['generalization_gap'].mean():.4f} ± {low_skew['generalization_gap'].std():.4f}")
    print(f"  T-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    # Test 2: Kolmogorov-Smirnov test for distribution differences
    ks_stat, ks_p = stats.ks_2samp(
        high_skew['ece'].dropna(),
        low_skew['ece'].dropna()
    )
    
    print(f"\nTest 2: ECE Distribution Difference (KS Test)")
    print(f"  KS statistic: {ks_stat:.4f}, p={ks_p:.4f}")
    
    # Test 3: ANOVA for lambda_range across regularization types
    # Group methods by regularization type
    reg_types = []
    for method in df_directional['method_name']:
        if 'dropout' in method:
            reg_types.append('dropout')
        elif 'weight_decay' in method:
            reg_types.append('weight_decay')
        elif 'augmentation' in method:
            reg_types.append('augmentation')
        elif 'label_smoothing' in method:
            reg_types.append('label_smoothing')
        elif 'sam' in method:
            reg_types.append('sam')
        elif 'igp' in method:
            reg_types.append('igp')
        else:
            reg_types.append('baseline')
    
    df_directional['reg_type'] = reg_types
    
    # One-way ANOVA
    groups = [group['lambda_range'].dropna() for name, group in df_directional.groupby('reg_type')]
    f_stat, anova_p = stats.f_oneway(*groups)
    
    print(f"\nTest 3: Lambda Range Across Regularization Types (ANOVA)")
    print(f"  F-statistic: {f_stat:.4f}, p={anova_p:.4f}")
    
    for reg_type in df_directional['reg_type'].unique():
        reg_data = df_directional[df_directional['reg_type'] == reg_type]
        print(f"  {reg_type:20s}: λ_range = {reg_data['lambda_range'].mean():.4f} ± {reg_data['lambda_range'].std():.4f}")
    
    # Test 4: Permutation test for lambda_skewness vs generalization_gap
    print(f"\nTest 4: Permutation Test (Lambda Skewness vs Gen Gap)")
    
    observed_corr, _ = stats.pearsonr(
        df_directional['lambda_skewness'].dropna(),
        df_directional['generalization_gap'].dropna()
    )
    
    # Permutation test
    n_permutations = 10000
    perm_corrs = []
    
    x = df_directional['lambda_skewness'].dropna().values
    y = df_directional['generalization_gap'].dropna().values
    
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        corr, _ = stats.pearsonr(x, y_perm)
        perm_corrs.append(corr)
    
    perm_p = np.mean(np.abs(perm_corrs) >= np.abs(observed_corr))
    
    print(f"  Observed correlation: {observed_corr:.4f}")
    print(f"  Permutation p-value:  {perm_p:.4f}")
    print(f"  (Based on {n_permutations} permutations)")


def cluster_methods_by_lambda_profile(df_directional, dataset_name, output_dir):
    """
    Cluster methods based on their lambda directional profiles.
    """
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} METHOD CLUSTERING")
    print(f"{'='*80}")
    
    # Features for clustering
    features = [
        'lambda_mean', 'lambda_std', 'lambda_range', 
        'lambda_skewness', 'lambda_kurtosis', 'lambda_cv'
    ]
    
    # Prepare data
    X = df_directional[features].values
    methods = df_directional['method_name'].values
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Hierarchical clustering
    linkage_matrix = linkage(X_scaled, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(14, 8))
    dendrogram(linkage_matrix, labels=methods, leaf_rotation=90, leaf_font_size=10)
    plt.title(f'{dataset_name.upper()} - Method Clustering by Lambda Profile', fontsize=14, fontweight='bold')
    plt.xlabel('Method')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_method_clustering_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved clustering dendrogram")
    
    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    
    # Color by ECE
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=df_directional['ece'], 
                         s=150, alpha=0.6, cmap='RdYlGn_r', edgecolors='black', linewidth=1)
    
    # Add labels
    for i, method in enumerate(methods):
        plt.annotate(method, (X_pca[i, 0], X_pca[i, 1]), 
                    fontsize=8, alpha=0.7, 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.colorbar(scatter, label='ECE (Calibration Error)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'{dataset_name.upper()} - Lambda Profile Space (PCA)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_lambda_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved PCA visualization")
    print(f"  PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
    print(f"  PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")


def analyze_lambda_percentiles(df_directional, dataset_name, output_dir):
    """
    Analyze if extreme lambda values (min/max) matter more than mean.
    """
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} EXTREME VALUE ANALYSIS")
    print(f"{'='*80}")
    
    # Correlation of min/max/percentiles with performance
    features = {
        'lambda_min': 'Flattest Direction',
        'lambda_max': 'Sharpest Direction',
        'lambda_q25': '25th Percentile',
        'lambda_q75': '75th Percentile',
        'lambda_median': 'Median'
    }
    
    print(f"\nCorrelations with Generalization Gap:")
    print("-" * 60)
    
    for feat, name in features.items():
        r, p = stats.pearsonr(
            df_directional[feat].dropna(),
            df_directional['generalization_gap'].dropna()
        )
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:25s}: r={r:7.4f}, p={p:.4f} {sig}")
    
    print(f"\nCorrelations with ECE:")
    print("-" * 60)
    
    for feat, name in features.items():
        r, p = stats.pearsonr(
            df_directional[feat].dropna(),
            df_directional['ece'].dropna()
        )
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:25s}: r={r:7.4f}, p={p:.4f} {sig}")


def main():
    """Main analysis routine."""
    parser = argparse.ArgumentParser(description='Advanced lambda analysis')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist', 'both'])
    parser.add_argument('--results-dir', type=str, default='results/proof_of_concept')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ADVANCED LAMBDA ANALYSIS")
    print("="*80)
    
    # Determine datasets
    datasets = []
    if args.dataset == 'both':
        for ds in ['cifar10', 'mnist']:
            if (results_dir / ds).exists():
                datasets.append(ds)
    else:
        if (results_dir / args.dataset).exists():
            datasets.append(args.dataset)
    
    for dataset in datasets:
        # Load directional statistics (from previous analysis)
        directional_file = analysis_dir / f'{dataset}_directional_statistics.csv'
        if not directional_file.exists():
            print(f"\nWarning: Run analyze_directional_lambdas.py first for {dataset}")
            continue
        
        df_directional = pd.read_csv(directional_file)
        
        # Load temporal data
        df_temporal = load_temporal_lambda_data(results_dir, dataset)
        
        # Run analyses
        temporal_df = analyze_temporal_dynamics(df_temporal, dataset, analysis_dir)
        
        statistical_hypothesis_tests(df_directional, None, dataset)
        
        cluster_methods_by_lambda_profile(df_directional, dataset, analysis_dir)
        
        analyze_lambda_percentiles(df_directional, dataset, analysis_dir)
    
    print(f"\n{'='*80}")
    print("ADVANCED ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
