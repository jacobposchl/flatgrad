"""
Temporal Lambda Dynamics Analysis

Investigates how lambda evolves during training and whether trajectory features
predict final model quality (test accuracy, generalization, calibration).

Focus areas:
1. Full lambda trajectories (all checkpoints)
2. Trajectory features (slope, AUC, inflection points)
3. Sign flip investigation (lambda-ECE relationship reversal)
4. Per-dataset temporal patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats, integrate
from typing import Dict, List, Tuple

# Configuration
RESULTS_DIR = Path(r"C:\Users\Jacob Poschl\Desktop\flatgrad\flatgrad\results\proof_of_concept")
OUTPUT_DIR = Path(r"C:\Users\Jacob Poschl\Desktop\flatgrad\flatgrad\results\proof_of_concept\analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_temporal_data() -> pd.DataFrame:
    """Load temporal lambda trajectories from all experiments."""
    results = []
    
    for experiment_dir in RESULTS_DIR.rglob('lambda_data.npz'):
        try:
            lambda_data = np.load(experiment_dir, allow_pickle=True)
            summary_path = experiment_dir.parent / 'summary.json'
            
            if not summary_path.exists():
                continue
                
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            # Extract temporal lambda data
            epochs = lambda_data['epochs']
            lambda_values_per_epoch = lambda_data['lambda_values_per_epoch']
            
            if len(epochs) == 0 or len(lambda_values_per_epoch) == 0:
                continue
            
            # Compute lambda metrics per epoch
            lambda_means = []
            lambda_stds = []
            
            for lambdas_at_epoch in lambda_values_per_epoch:
                if lambdas_at_epoch is not None and len(lambdas_at_epoch) > 0:
                    lambda_means.append(np.mean(lambdas_at_epoch))
                    lambda_stds.append(np.std(lambdas_at_epoch))
                else:
                    lambda_means.append(np.nan)
                    lambda_stds.append(np.nan)
            
            results.append({
                'method': summary['method_name'],
                'dataset': summary['dataset'],
                'epochs': epochs,
                'lambda_mean': lambda_means,
                'lambda_std': lambda_stds,
                'final_test_acc': summary['final_test_accuracy'],
                'final_ece': summary.get('final_ece', np.nan),
                'final_gen_gap': summary.get('final_generalization_gap', np.nan)
            })
            
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

def compute_trajectory_features(epochs: np.ndarray, lambda_values: np.ndarray) -> Dict:
    """Extract features from lambda trajectory."""
    # Remove NaN values
    valid_mask = ~np.isnan(lambda_values)
    epochs_clean = np.array(epochs)[valid_mask]
    lambda_clean = lambda_values[valid_mask]
    
    if len(lambda_clean) < 3:
        return {
            'slope': np.nan,
            'auc': np.nan,
            'initial': np.nan,
            'final': np.nan,
            'change': np.nan,
            'volatility': np.nan,
            'time_to_converge': np.nan
        }
    
    # Linear trend (slope)
    slope, intercept, r_value, p_value, std_err = stats.linregress(epochs_clean, lambda_clean)
    
    # Area under curve (normalized by time)
    auc = integrate.trapz(lambda_clean, epochs_clean) / (epochs_clean[-1] - epochs_clean[0])
    
    # Basic statistics
    initial = lambda_clean[0]
    final = lambda_clean[-1]
    change = final - initial
    
    # Volatility (standard deviation of differences)
    volatility = np.std(np.diff(lambda_clean))
    
    # Time to converge (when lambda stabilizes within 10% of final)
    threshold = 0.1 * abs(final)
    converged_mask = np.abs(lambda_clean - final) < threshold
    if np.any(converged_mask):
        time_to_converge = epochs_clean[np.argmax(converged_mask)]
    else:
        time_to_converge = epochs_clean[-1]
    
    return {
        'slope': slope,
        'auc': auc,
        'initial': initial,
        'final': final,
        'change': change,
        'volatility': volatility,
        'time_to_converge': time_to_converge
    }

def analyze_trajectories(df: pd.DataFrame):
    """Extract and analyze trajectory features."""
    print("\n" + "="*80)
    print("TEMPORAL TRAJECTORY ANALYSIS")
    print("="*80)
    
    if len(df) == 0:
        print("No data to analyze")
        return pd.DataFrame()
    
    print(f"Processing {len(df)} experiments")
    
    # Compute features for each experiment
    trajectory_data = []
    
    for _, row in df.iterrows():
        features_mean = compute_trajectory_features(row['epochs'], np.array(row['lambda_mean']))
        features_std = compute_trajectory_features(row['epochs'], np.array(row['lambda_std']))
        
        trajectory_data.append({
            'method': row['method'],
            'dataset': row['dataset'],
            'final_test_acc': row['final_test_acc'],
            'final_ece': row['final_ece'],
            'final_gen_gap': row['final_gen_gap'],
            'lambda_mean_slope': features_mean['slope'],
            'lambda_mean_auc': features_mean['auc'],
            'lambda_mean_change': features_mean['change'],
            'lambda_mean_volatility': features_mean['volatility'],
            'lambda_std_slope': features_std['slope'],
            'lambda_std_auc': features_std['auc'],
            'lambda_std_change': features_std['change'],
            'lambda_std_volatility': features_std['volatility']
        })
    
    if len(trajectory_data) == 0:
        print("No trajectory data found")
        return pd.DataFrame()
    
    traj_df = pd.DataFrame(trajectory_data)
    
    # Correlation analysis per dataset
    print("\n1. Trajectory Features vs Performance (per-dataset):\n")
    
    for dataset in ['mnist', 'cifar10']:
        df_dataset = traj_df[traj_df['dataset'] == dataset]
        print(f"{dataset.upper()}:")
        
        # Test each trajectory feature
        features = [
            ('lambda_mean_slope', 'λ_mean slope'),
            ('lambda_mean_auc', 'λ_mean AUC'),
            ('lambda_mean_change', 'λ_mean change'),
            ('lambda_std_slope', 'λ_std slope'),
            ('lambda_std_change', 'λ_std change')
        ]
        
        best_ece = {'r': 0, 'feature': ''}
        
        for feat_col, feat_name in features:
            # Correlation with ECE
            valid_data = df_dataset[[feat_col, 'final_ece']].dropna()
            if len(valid_data) >= 3:
                r, p = stats.pearsonr(valid_data[feat_col], valid_data['final_ece'])
                if abs(r) > abs(best_ece['r']):
                    best_ece = {'r': r, 'p': p, 'feature': feat_name}
        
        if best_ece['feature']:
            sig = '***' if best_ece['p'] < 0.001 else '**' if best_ece['p'] < 0.01 else '*' if best_ece['p'] < 0.05 else 'ns'
            print(f"  Best ECE predictor: {best_ece['feature']:<20} r={best_ece['r']:+.4f}, p={best_ece['p']:.4f} {sig}")
        print()
    
    return traj_df

def investigate_sign_flip(df: pd.DataFrame, traj_df: pd.DataFrame):
    """Investigate why lambda-ECE correlation flips sign between datasets."""
    print("\n2. Sign Flip Investigation:\n")
    
    # Check if accuracy explains the flip
    for dataset in ['mnist', 'cifar10']:
        df_dataset = df[df['dataset'] == dataset]
        
        # Get a sample trajectory to compute mean lambda
        sample = df_dataset.iloc[0]
        mean_lambda_final = np.nanmean(sample['lambda_mean'][-5:])  # Average last 5 epochs
        
        ece_range = (df_dataset['final_ece'].min(), df_dataset['final_ece'].max())
        acc_range = (df_dataset['final_test_acc'].min(), df_dataset['final_test_acc'].max())
        
        print(f"{dataset.upper()}:")
        print(f"  Accuracy range:  {acc_range[0]:.4f} - {acc_range[1]:.4f}")
        print(f"  ECE range:       {ece_range[0]:.4f} - {ece_range[1]:.4f}")
        print(f"  Typical λ_mean:  {mean_lambda_final:.4f}")
    
    print("\nHypothesis: Sign flip may be due to:")
    print("  - Different operating regimes (near-perfect vs struggling)")
    print("  - Different ECE scales (0.01-0.02 vs 0.06-0.36)")
    print("  - Nonlinear relationship with inflection point\n")

def create_visualizations(df: pd.DataFrame, traj_df: pd.DataFrame):
    """Create temporal trajectory visualizations."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = {'mnist': '#e74c3c', 'cifar10': '#9b59b6'}
    
    # Plot 1: Lambda_mean trajectories (best vs worst per dataset)
    ax1 = fig.add_subplot(gs[0, :2])
    
    for dataset in ['mnist', 'cifar10']:
        df_dataset = df[df['dataset'] == dataset]
        
        # Best (highest accuracy)
        best_idx = df_dataset['final_test_acc'].idxmax()
        best = df_dataset.loc[best_idx]
        ax1.plot(best['epochs'], best['lambda_mean'], color=colors[dataset], 
                linewidth=2.5, label=f"{dataset.upper()} best ({best['method']})", alpha=0.9)
        
        # Worst (lowest accuracy)
        worst_idx = df_dataset['final_test_acc'].idxmin()
        worst = df_dataset.loc[worst_idx]
        ax1.plot(worst['epochs'], worst['lambda_mean'], color=colors[dataset], 
                linewidth=2.5, linestyle='--', label=f"{dataset.upper()} worst ({worst['method']})", alpha=0.6)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Lambda Mean', fontsize=12, fontweight='bold')
    ax1.set_title('A. Lambda Evolution: Best vs Worst Models', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(alpha=0.3)
    
    # Plot 2: All trajectories (spaghetti plot)
    ax2 = fig.add_subplot(gs[0, 2])
    
    for _, row in df.iterrows():
        ax2.plot(row['epochs'], row['lambda_mean'], 
                color=colors[row['dataset']], alpha=0.2, linewidth=1)
    
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Lambda Mean', fontsize=11, fontweight='bold')
    ax2.set_title('B. All Trajectories', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Plot 3: Lambda slope vs ECE (per-dataset)
    ax3 = fig.add_subplot(gs[1, 0])
    
    for dataset in ['mnist', 'cifar10']:
        subset = traj_df[traj_df['dataset'] == dataset]
        valid = subset[['lambda_mean_slope', 'final_ece']].dropna()
        
        if len(valid) >= 3:
            ax3.scatter(valid['lambda_mean_slope'], valid['final_ece'],
                       c=colors[dataset], label=dataset.upper(), alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
            
            r, p = stats.pearsonr(valid['lambda_mean_slope'], valid['final_ece'])
            y_pos = 0.95 if dataset == 'mnist' else 0.85
            ax3.text(0.05, y_pos, f'{dataset.upper()}: r={r:+.3f}', 
                    transform=ax3.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor=colors[dataset], alpha=0.2))
    
    ax3.set_xlabel('λ_mean Slope\n(rate of change)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Final ECE', fontsize=11, fontweight='bold')
    ax3.set_title('C. Trajectory Slope vs ECE', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # Plot 4: Lambda AUC vs ECE
    ax4 = fig.add_subplot(gs[1, 1])
    
    for dataset in ['mnist', 'cifar10']:
        subset = traj_df[traj_df['dataset'] == dataset]
        valid = subset[['lambda_mean_auc', 'final_ece']].dropna()
        
        if len(valid) >= 3:
            ax4.scatter(valid['lambda_mean_auc'], valid['final_ece'],
                       c=colors[dataset], label=dataset.upper(), alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
            
            r, p = stats.pearsonr(valid['lambda_mean_auc'], valid['final_ece'])
            y_pos = 0.95 if dataset == 'mnist' else 0.85
            ax4.text(0.05, y_pos, f'{dataset.upper()}: r={r:+.3f}', 
                    transform=ax4.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor=colors[dataset], alpha=0.2))
    
    ax4.set_xlabel('λ_mean AUC\n(integrated flatness)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Final ECE', fontsize=11, fontweight='bold')
    ax4.set_title('D. Integrated Flatness vs ECE', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # Plot 5: Lambda_std trajectories
    ax5 = fig.add_subplot(gs[1, 2])
    
    for dataset in ['mnist', 'cifar10']:
        df_dataset = df[df['dataset'] == dataset]
        best_idx = df_dataset['final_test_acc'].idxmax()
        best = df_dataset.loc[best_idx]
        ax5.plot(best['epochs'], best['lambda_std'], color=colors[dataset], 
                linewidth=2.5, label=f"{dataset.upper()} best", alpha=0.9)
    
    ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Lambda Std', fontsize=11, fontweight='bold')
    ax5.set_title('E. Directional Variance\nEvolution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    
    # Plot 6: Lambda mean colored by accuracy
    ax6 = fig.add_subplot(gs[2, :])
    
    for dataset in ['mnist', 'cifar10']:
        subset = traj_df[traj_df['dataset'] == dataset]
        valid = subset[['lambda_mean_auc', 'final_ece', 'final_test_acc']].dropna()
        
        if len(valid) >= 3:
            scatter = ax6.scatter(valid['lambda_mean_auc'], valid['final_ece'],
                                 c=valid['final_test_acc'], cmap='RdYlGn', 
                                 s=150, alpha=0.8, edgecolors='black', linewidth=1,
                                 vmin=0.5, vmax=1.0)
    
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Test Accuracy', fontsize=11, fontweight='bold')
    
    ax6.set_xlabel('Lambda Mean AUC', fontsize=12, fontweight='bold')
    ax6.set_ylabel('ECE', fontsize=12, fontweight='bold')
    ax6.set_title('F. Lambda-ECE Relationship Colored by Accuracy (Sign Flip Investigation)', fontsize=13, fontweight='bold')
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'temporal_dynamics_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}\n")
    plt.close()

def main():
    print("\nTemporal Lambda Dynamics Analysis")
    print("="*80)
    
    # Load data
    df = load_temporal_data()
    print(f"✓ Loaded {len(df)} experiments with temporal trajectories")
    
    if len(df) == 0:
        print("❌ No temporal data found. Check that experiments have lambda_data.npz files.")
        return
    
    # Analyze trajectories
    traj_df = analyze_trajectories(df)
    
    if len(traj_df) == 0:
        print("❌ No trajectory features computed")
        return
    
    # Investigate sign flip
    investigate_sign_flip(df, traj_df)
    
    # Visualizations
    create_visualizations(df, traj_df)
    
    # Save trajectory features
    output_path = OUTPUT_DIR / 'trajectory_features.csv'
    traj_df.to_csv(output_path, index=False)
    print(f"✓ Saved trajectory features: {output_path}")

if __name__ == '__main__':
    main()
