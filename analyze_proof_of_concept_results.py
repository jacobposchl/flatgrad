"""
Comprehensive analysis of proof-of-concept experiment results.

Analyzes whether λ (lambda/curvature rate) correlates with desirable model properties:
- Generalization (test accuracy, generalization gap)
- Calibration (ECE)
- Overfitting resistance

This script answers: Is lambda a scientifically meaningful diagnostic metric?
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# Configuration
RESULTS_DIR = Path(r"C:\Users\Jacob Poschl\Desktop\flatgrad\flatgrad\results\proof_of_concept")
OUTPUT_DIR = Path(r"C:\Users\Jacob Poschl\Desktop\flatgrad\flatgrad\results\proof_of_concept\analysis")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def print_section(title: str, level: int = 1):
    """Print formatted section header."""
    if level == 1:
        print(f"\n{'='*100}")
        print(f"{title}")
        print('='*100)
    elif level == 2:
        print(f"\n{'-'*80}")
        print(f"{title}")
        print('-'*80)
    else:
        print(f"\n{title}")

def load_all_results() -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    print_section("1. LOADING EXPERIMENT RESULTS", level=1)
    
    results = []
    for summary_file in RESULTS_DIR.rglob('summary.json'):
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"⚠️  Failed to load {summary_file}: {e}")
    
    df = pd.DataFrame(results)
    
    print(f"✓ Loaded {len(df)} experiments")
    print(f"\nDatasets: {df['dataset'].unique().tolist()}")
    print(f"Methods: {df['method_name'].unique().tolist()}")
    
    return df

def compute_correlation_analysis(df: pd.DataFrame) -> Dict:
    """Compute correlations between lambda and key metrics."""
    print_section("2. CORRELATION ANALYSIS: Lambda vs. Model Quality", level=1)
    
    # Metrics to correlate with lambda
    metrics = {
        'Test Accuracy': 'final_test_accuracy',
        'Generalization Gap': 'final_generalization_gap',
        'ECE (Calibration)': 'final_ece',
        'Train Accuracy': 'final_train_accuracy',
    }
    
    results = {}
    
    for metric_name, metric_col in metrics.items():
        if metric_col not in df.columns or 'final_lambda_mean' not in df.columns:
            continue
        
        # Remove NaN values
        valid_data = df[[metric_col, 'final_lambda_mean']].dropna()
        
        if len(valid_data) < 3:
            print(f"\n⚠️  Insufficient data for {metric_name}")
            continue
        
        # Compute correlation
        pearson_r, pearson_p = stats.pearsonr(valid_data['final_lambda_mean'], valid_data[metric_col])
        spearman_r, spearman_p = stats.spearmanr(valid_data['final_lambda_mean'], valid_data[metric_col])
        
        results[metric_name] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_samples': len(valid_data)
        }
        
        # Print results
        print(f"\n{metric_name}:")
        print(f"  Pearson correlation:  r = {pearson_r:+.4f}, p = {pearson_p:.4f} {'***' if pearson_p < 0.001 else '**' if pearson_p < 0.01 else '*' if pearson_p < 0.05 else 'ns'}")
        print(f"  Spearman correlation: ρ = {spearman_r:+.4f}, p = {spearman_p:.4f} {'***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'ns'}")
        print(f"  Sample size: n = {len(valid_data)}")
        
        # Interpretation
        if pearson_p < 0.05:
            direction = "positive" if pearson_r > 0 else "negative"
            strength = "strong" if abs(pearson_r) > 0.7 else "moderate" if abs(pearson_r) > 0.4 else "weak"
            print(f"  → {strength.upper()} {direction} correlation (statistically significant)")
        else:
            print(f"  → No significant correlation")
    
    return results

def analyze_by_dataset(df: pd.DataFrame):
    """Analyze results separately for each dataset."""
    print_section("3. DATASET-SPECIFIC ANALYSIS", level=1)
    
    for dataset in df['dataset'].unique():
        print_section(f"Dataset: {dataset.upper()}", level=2)
        
        df_subset = df[df['dataset'] == dataset].copy()
        
        # Sort by lambda
        df_subset = df_subset.sort_values('final_lambda_mean')
        
        print(f"\nResults sorted by Lambda (low to high):")
        print(f"{'Method':<25} {'Lambda':>10} {'Test Acc':>10} {'Gen Gap':>10} {'ECE':>10}")
        print('-'*70)
        
        for _, row in df_subset.iterrows():
            print(f"{row['method_name']:<25} {row['final_lambda_mean']:>10.4f} "
                  f"{row['final_test_accuracy']:>10.4f} {row['final_generalization_gap']:>10.4f} "
                  f"{row['final_ece']:>10.4f}")
        
        # Find best models
        best_acc_idx = df_subset['final_test_accuracy'].idxmax()
        best_gen_idx = df_subset['final_generalization_gap'].abs().idxmin()
        best_ece_idx = df_subset['final_ece'].idxmin()
        
        print(f"\n🏆 Best Models:")
        print(f"  Highest test accuracy: {df_subset.loc[best_acc_idx, 'method_name']} "
              f"(λ={df_subset.loc[best_acc_idx, 'final_lambda_mean']:.4f}, "
              f"acc={df_subset.loc[best_acc_idx, 'final_test_accuracy']:.4f})")
        print(f"  Best generalization:   {df_subset.loc[best_gen_idx, 'method_name']} "
              f"(λ={df_subset.loc[best_gen_idx, 'final_lambda_mean']:.4f}, "
              f"gap={df_subset.loc[best_gen_idx, 'final_generalization_gap']:.4f})")
        print(f"  Best calibration:      {df_subset.loc[best_ece_idx, 'method_name']} "
              f"(λ={df_subset.loc[best_ece_idx, 'final_lambda_mean']:.4f}, "
              f"ECE={df_subset.loc[best_ece_idx, 'final_ece']:.4f})")

def analyze_regularization_effects(df: pd.DataFrame):
    """Analyze how different regularization methods affect lambda."""
    print_section("4. REGULARIZATION METHOD ANALYSIS", level=1)
    
    # Group methods by type
    method_groups = {
        'Baseline': ['baseline'],
        'Dropout': [m for m in df['method_name'].unique() if 'dropout' in m],
        'Weight Decay': [m for m in df['method_name'].unique() if 'weight_decay' in m],
        'Label Smoothing': [m for m in df['method_name'].unique() if 'label_smoothing' in m],
        'SAM': [m for m in df['method_name'].unique() if 'sam' in m],
        'IGP': [m for m in df['method_name'].unique() if 'igp' in m],
        'Augmentation': [m for m in df['method_name'].unique() if 'augmentation' in m],
    }
    
    for group_name, methods in method_groups.items():
        if not methods:
            continue
        
        print(f"\n{group_name}:")
        group_data = df[df['method_name'].isin(methods)]
        
        if len(group_data) == 0:
            continue
        
        print(f"  Methods: {', '.join(methods)}")
        print(f"  Lambda range: [{group_data['final_lambda_mean'].min():.4f}, "
              f"{group_data['final_lambda_mean'].max():.4f}]")
        print(f"  Avg test accuracy: {group_data['final_test_accuracy'].mean():.4f} ± "
              f"{group_data['final_test_accuracy'].std():.4f}")
        print(f"  Avg generalization gap: {group_data['final_generalization_gap'].mean():.4f} ± "
              f"{group_data['final_generalization_gap'].std():.4f}")
        print(f"  Avg ECE: {group_data['final_ece'].mean():.4f} ± "
              f"{group_data['final_ece'].std():.4f}")

def test_lambda_hypotheses(df: pd.DataFrame):
    """Test specific hypotheses about lambda."""
    print_section("5. HYPOTHESIS TESTING", level=1)
    
    # Hypothesis 1: Higher lambda correlates with better generalization
    print("\n📊 Hypothesis 1: Higher λ → Better generalization (smaller gap)")
    valid_data = df[['final_lambda_mean', 'final_generalization_gap']].dropna()
    
    if len(valid_data) >= 3:
        r, p = stats.pearsonr(valid_data['final_lambda_mean'], 
                              valid_data['final_generalization_gap'].abs())
        
        print(f"  Correlation (λ vs |gen_gap|): r = {r:.4f}, p = {p:.4f}")
        
        if p < 0.05 and r < 0:
            print(f"  ✅ SUPPORTED: Higher λ correlates with smaller generalization gap")
        elif p < 0.05 and r > 0:
            print(f"  ❌ REJECTED: Higher λ correlates with LARGER generalization gap")
        else:
            print(f"  ⚠️  INCONCLUSIVE: No significant correlation found")
    
    # Hypothesis 2: Higher lambda correlates with better calibration
    print("\n📊 Hypothesis 2: Higher λ → Better calibration (lower ECE)")
    valid_data = df[['final_lambda_mean', 'final_ece']].dropna()
    
    if len(valid_data) >= 3:
        r, p = stats.pearsonr(valid_data['final_lambda_mean'], valid_data['final_ece'])
        
        print(f"  Correlation (λ vs ECE): r = {r:.4f}, p = {p:.4f}")
        
        if p < 0.05 and r < 0:
            print(f"  ✅ SUPPORTED: Higher λ correlates with better calibration")
        elif p < 0.05 and r > 0:
            print(f"  ❌ REJECTED: Higher λ correlates with WORSE calibration")
        else:
            print(f"  ⚠️  INCONCLUSIVE: No significant correlation found")
    
    # Hypothesis 3: Higher lambda correlates with higher test accuracy
    print("\n📊 Hypothesis 3: Higher λ → Higher test accuracy")
    valid_data = df[['final_lambda_mean', 'final_test_accuracy']].dropna()
    
    if len(valid_data) >= 3:
        r, p = stats.pearsonr(valid_data['final_lambda_mean'], valid_data['final_test_accuracy'])
        
        print(f"  Correlation (λ vs test_acc): r = {r:.4f}, p = {p:.4f}")
        
        if p < 0.05 and r > 0:
            print(f"  ✅ SUPPORTED: Higher λ correlates with higher test accuracy")
        elif p < 0.05 and r < 0:
            print(f"  ❌ REJECTED: Higher λ correlates with LOWER test accuracy")
        else:
            print(f"  ⚠️  INCONCLUSIVE: No significant correlation found")
    
    # Hypothesis 4: Regularization increases lambda
    print("\n📊 Hypothesis 4: Regularization increases λ (vs baseline)")
    
    baseline_lambda = df[df['method_name'] == 'baseline']['final_lambda_mean'].values
    regularized_lambda = df[df['method_name'] != 'baseline']['final_lambda_mean'].dropna().values
    
    if len(baseline_lambda) > 0 and len(regularized_lambda) > 0:
        baseline_mean = baseline_lambda.mean()
        regularized_mean = regularized_lambda.mean()
        
        # T-test
        t_stat, p_value = stats.ttest_ind(regularized_lambda, baseline_lambda)
        
        print(f"  Baseline λ:     {baseline_mean:.4f} (n={len(baseline_lambda)})")
        print(f"  Regularized λ:  {regularized_mean:.4f} (n={len(regularized_lambda)})")
        print(f"  Difference:     {regularized_mean - baseline_mean:+.4f}")
        print(f"  T-test:         t = {t_stat:.4f}, p = {p_value:.4f}")
        
        if p_value < 0.05 and regularized_mean > baseline_mean:
            print(f"  ✅ SUPPORTED: Regularization significantly increases λ")
        elif p_value < 0.05 and regularized_mean < baseline_mean:
            print(f"  ❌ REJECTED: Regularization significantly DECREASES λ")
        else:
            print(f"  ⚠️  INCONCLUSIVE: No significant difference")

def create_visualizations(df: pd.DataFrame):
    """Create comprehensive visualizations."""
    print_section("6. CREATING VISUALIZATIONS", level=1)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # Figure 1: Lambda vs. Key Metrics (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Subplot 1: Lambda vs Test Accuracy
    for dataset in df['dataset'].unique():
        df_subset = df[df['dataset'] == dataset]
        axes[0, 0].scatter(df_subset['final_lambda_mean'], df_subset['final_test_accuracy'],
                          label=dataset.upper(), alpha=0.7, s=100)
    
    axes[0, 0].set_xlabel('Lambda Mean', fontsize=12)
    axes[0, 0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0, 0].set_title('Lambda vs. Test Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Add correlation line
    valid_data = df[['final_lambda_mean', 'final_test_accuracy']].dropna()
    if len(valid_data) >= 2:
        z = np.polyfit(valid_data['final_lambda_mean'], valid_data['final_test_accuracy'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_data['final_lambda_mean'].min(), 
                            valid_data['final_lambda_mean'].max(), 100)
        axes[0, 0].plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label='Trend')
    
    # Subplot 2: Lambda vs Generalization Gap
    for dataset in df['dataset'].unique():
        df_subset = df[df['dataset'] == dataset]
        axes[0, 1].scatter(df_subset['final_lambda_mean'], df_subset['final_generalization_gap'],
                          label=dataset.upper(), alpha=0.7, s=100)
    
    axes[0, 1].set_xlabel('Lambda Mean', fontsize=12)
    axes[0, 1].set_ylabel('Generalization Gap', fontsize=12)
    axes[0, 1].set_title('Lambda vs. Generalization Gap', fontsize=14, fontweight='bold')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Subplot 3: Lambda vs ECE
    for dataset in df['dataset'].unique():
        df_subset = df[df['dataset'] == dataset]
        axes[1, 0].scatter(df_subset['final_lambda_mean'], df_subset['final_ece'],
                          label=dataset.upper(), alpha=0.7, s=100)
    
    axes[1, 0].set_xlabel('Lambda Mean', fontsize=12)
    axes[1, 0].set_ylabel('ECE (Calibration Error)', fontsize=12)
    axes[1, 0].set_title('Lambda vs. Calibration (ECE)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Subplot 4: Lambda by Regularization Method
    method_order = df.groupby('method_name')['final_lambda_mean'].mean().sort_values().index
    sns.boxplot(data=df, y='method_name', x='final_lambda_mean', order=method_order, ax=axes[1, 1])
    axes[1, 1].set_xlabel('Lambda Mean', fontsize=12)
    axes[1, 1].set_ylabel('Regularization Method', fontsize=12)
    axes[1, 1].set_title('Lambda Distribution by Method', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'lambda_correlations.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Figure 2: Heatmap of correlations
    fig, ax = plt.subplots(figsize=(10, 8))
    
    correlation_metrics = ['final_lambda_mean', 'final_lambda_std', 'final_test_accuracy', 
                          'final_train_accuracy', 'final_generalization_gap', 'final_ece']
    
    corr_matrix = df[correlation_metrics].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)
    
    ax.set_title('Correlation Matrix: Lambda and Performance Metrics', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'correlation_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Figure 3: Method comparison (grouped bar chart)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    df_sorted = df.sort_values('final_lambda_mean')
    
    # Lambda
    axes[0].barh(df_sorted['method_name'], df_sorted['final_lambda_mean'], color='steelblue')
    axes[0].set_xlabel('Lambda Mean', fontsize=12)
    axes[0].set_title('Lambda by Method', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='x')
    
    # Test Accuracy
    axes[1].barh(df_sorted['method_name'], df_sorted['final_test_accuracy'], color='forestgreen')
    axes[1].set_xlabel('Test Accuracy', fontsize=12)
    axes[1].set_title('Test Accuracy by Method', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='x')
    
    # Generalization Gap
    colors = ['red' if x > 0 else 'green' for x in df_sorted['final_generalization_gap']]
    axes[2].barh(df_sorted['method_name'], df_sorted['final_generalization_gap'], color=colors, alpha=0.7)
    axes[2].axvline(x=0, color='k', linestyle='--', linewidth=1)
    axes[2].set_xlabel('Generalization Gap', fontsize=12)
    axes[2].set_title('Generalization Gap by Method', fontsize=14, fontweight='bold')
    axes[2].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'method_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def generate_summary_report(df: pd.DataFrame, correlations: Dict):
    """Generate a text summary report."""
    print_section("7. SCIENTIFIC FINDINGS SUMMARY", level=1)
    
    print("\n" + "="*100)
    print("PROOF OF CONCEPT: IS LAMBDA A MEANINGFUL DIAGNOSTIC METRIC?")
    print("="*100)
    
    print("\n📊 EXPERIMENT SCOPE:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Datasets: {', '.join(df['dataset'].unique())}")
    print(f"  Regularization methods: {len(df['method_name'].unique())}")
    
    print("\n🔬 KEY FINDINGS:")
    
    # Finding 1: Correlation with generalization
    if 'Generalization Gap' in correlations:
        gen_corr = correlations['Generalization Gap']
        if gen_corr['pearson_p'] < 0.05:
            direction = "POSITIVE" if gen_corr['pearson_r'] > 0 else "NEGATIVE"
            strength = abs(gen_corr['pearson_r'])
            print(f"\n  1. Lambda and Generalization:")
            print(f"     {direction} correlation (r={gen_corr['pearson_r']:.3f}, p={gen_corr['pearson_p']:.4f})")
            if gen_corr['pearson_r'] < 0:
                print(f"     → Higher λ correlates with BETTER generalization (smaller gap)")
                print(f"     → SCIENTIFICALLY MEANINGFUL ✅")
            else:
                print(f"     → Higher λ correlates with WORSE generalization")
                print(f"     → Unexpected result - investigate further ⚠️")
        else:
            print(f"\n  1. Lambda and Generalization:")
            print(f"     No significant correlation found (p={gen_corr['pearson_p']:.4f})")
            print(f"     → Lambda may not predict generalization ❌")
    
    # Finding 2: Correlation with calibration
    if 'ECE (Calibration)' in correlations:
        ece_corr = correlations['ECE (Calibration)']
        if ece_corr['pearson_p'] < 0.05:
            direction = "POSITIVE" if ece_corr['pearson_r'] > 0 else "NEGATIVE"
            print(f"\n  2. Lambda and Calibration:")
            print(f"     {direction} correlation (r={ece_corr['pearson_r']:.3f}, p={ece_corr['pearson_p']:.4f})")
            if ece_corr['pearson_r'] < 0:
                print(f"     → Higher λ correlates with BETTER calibration (lower ECE)")
                print(f"     → SCIENTIFICALLY MEANINGFUL ✅")
            else:
                print(f"     → Higher λ correlates with WORSE calibration")
                print(f"     → Unexpected result - investigate further ⚠️")
        else:
            print(f"\n  2. Lambda and Calibration:")
            print(f"     No significant correlation found (p={ece_corr['pearson_p']:.4f})")
            print(f"     → Lambda may not predict calibration ❌")
    
    # Finding 3: Correlation with test accuracy
    if 'Test Accuracy' in correlations:
        acc_corr = correlations['Test Accuracy']
        if acc_corr['pearson_p'] < 0.05:
            direction = "POSITIVE" if acc_corr['pearson_r'] > 0 else "NEGATIVE"
            print(f"\n  3. Lambda and Test Accuracy:")
            print(f"     {direction} correlation (r={acc_corr['pearson_r']:.3f}, p={acc_corr['pearson_p']:.4f})")
            if acc_corr['pearson_r'] > 0:
                print(f"     → Higher λ correlates with higher accuracy")
                print(f"     → SCIENTIFICALLY MEANINGFUL ✅")
        else:
            print(f"\n  3. Lambda and Test Accuracy:")
            print(f"     No significant correlation found (p={acc_corr['pearson_p']:.4f})")
    
    # Finding 4: Effect of regularization
    baseline = df[df['method_name'] == 'baseline']
    regularized = df[df['method_name'] != 'baseline']
    
    if len(baseline) > 0 and len(regularized) > 0:
        baseline_lambda = baseline['final_lambda_mean'].mean()
        reg_lambda = regularized['final_lambda_mean'].mean()
        
        print(f"\n  4. Regularization Effect on Lambda:")
        print(f"     Baseline λ: {baseline_lambda:.4f}")
        print(f"     Regularized λ: {reg_lambda:.4f}")
        print(f"     Difference: {reg_lambda - baseline_lambda:+.4f}")
        
        if reg_lambda > baseline_lambda:
            print(f"     → Regularization INCREASES λ (flatter minima)")
            print(f"     → Consistent with theory ✅")
        else:
            print(f"     → Regularization DECREASES λ (sharper minima)")
            print(f"     → Inconsistent with theory ⚠️")
    
    print("\n" + "="*100)
    print("CONCLUSION:")
    print("="*100)
    
    # Overall verdict
    significant_correlations = sum(1 for corr in correlations.values() 
                                   if corr['pearson_p'] < 0.05)
    total_correlations = len(correlations)
    
    print(f"\nSignificant correlations found: {significant_correlations}/{total_correlations}")
    
    if significant_correlations >= 2:
        print("\n✅ LAMBDA APPEARS TO BE A SCIENTIFICALLY MEANINGFUL DIAGNOSTIC METRIC")
        print("\nLambda shows statistically significant correlations with multiple")
        print("desirable model properties, suggesting it captures meaningful information")
        print("about training quality and model behavior.")
    elif significant_correlations == 1:
        print("\n⚠️  LAMBDA SHOWS LIMITED DIAGNOSTIC VALUE")
        print("\nWhile lambda correlates with some metrics, the evidence is not strong")
        print("enough to conclusively establish it as a universal diagnostic.")
    else:
        print("\n❌ LAMBDA DOES NOT APPEAR TO BE A MEANINGFUL DIAGNOSTIC METRIC")
        print("\nNo significant correlations found with key performance metrics.")
        print("Lambda may not provide actionable insights for model training.")
    
    print("\n📈 RECOMMENDATIONS:")
    print("  1. Review correlation plots in the analysis/ folder")
    print("  2. Examine whether trends hold within each dataset separately")
    print("  3. Consider running more experiments with diverse hyperparameters")
    print("  4. Validate findings on additional datasets/architectures")
    
    print("\n" + "="*100 + "\n")

def save_summary_csv(df: pd.DataFrame, correlations: Dict):
    """Save analysis summary to CSV."""
    # Save full results
    output_path = OUTPUT_DIR / 'all_results.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Saved full results: {output_path}")
    
    # Save correlation summary
    corr_df = pd.DataFrame([
        {
            'Metric': name,
            'Pearson_r': data['pearson_r'],
            'Pearson_p': data['pearson_p'],
            'Spearman_r': data['spearman_r'],
            'Spearman_p': data['spearman_p'],
            'Significant': 'Yes' if data['pearson_p'] < 0.05 else 'No',
            'N_samples': data['n_samples']
        }
        for name, data in correlations.items()
    ])
    
    output_path = OUTPUT_DIR / 'correlation_summary.csv'
    corr_df.to_csv(output_path, index=False)
    print(f"✓ Saved correlation summary: {output_path}")

def main():
    """Main analysis pipeline."""
    print("\n" + "#"*100)
    print("PROOF OF CONCEPT: COMPREHENSIVE RESULTS ANALYSIS")
    print("Analyzing whether λ (lambda) is a scientifically meaningful diagnostic metric")
    print("#"*100)
    
    # Load data
    df = load_all_results()
    
    if len(df) == 0:
        print("\n❌ ERROR: No experiment results found!")
        print(f"Check that results exist in: {RESULTS_DIR}")
        return
    
    # Run analyses
    correlations = compute_correlation_analysis(df)
    analyze_by_dataset(df)
    analyze_regularization_effects(df)
    test_lambda_hypotheses(df)
    create_visualizations(df)
    save_summary_csv(df, correlations)
    generate_summary_report(df, correlations)
    
    print(f"\n✅ Analysis complete! Results saved to: {OUTPUT_DIR}")
    print(f"   - View plots: {OUTPUT_DIR / 'lambda_correlations.png'}")
    print(f"   - View heatmap: {OUTPUT_DIR / 'correlation_heatmap.png'}")
    print(f"   - View CSV: {OUTPUT_DIR / 'all_results.csv'}\n")

if __name__ == '__main__':
    main()
