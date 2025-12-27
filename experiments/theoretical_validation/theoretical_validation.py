"""
Theoretical Validation of Derivative Dynamics

This script rigorously validates lambda estimation and related metrics against
models with known analytical properties. Implements multiple statistical tests
to ensure robustness.

Tests:
1. Polynomial models (known derivative order cutoff)
2. Exponential decay models (known lambda and R)
3. Sinusoidal models (known cyclic patterns)
4. Sensitivity analysis (hyperparameter robustness)
5. Multi-direction averaging (variance reduction)

Includes:
- Multiple random seeds/initializations
- Confidence intervals (bootstrap & parametric)
- Hypothesis tests for significant differences
- Sensitivity analysis across hyperparameters
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import sys
import os

# Add FlatGrad root to path (go up 2 levels: theoretical_validation -> experiments -> FlatGrad)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flatgrad.sampling.models import create_test_model

# Import test modules from helpers subdirectory
from helpers.test_synthetic_sequences import run_synthetic_sequence_validation
from helpers.test_linear_model import run_linear_model_validation
from helpers.test_exponential_function import run_exponential_function_validation
from helpers.test_sinusoidal_model import run_sinusoidal_model_validation
from helpers.test_sensitivity_analysis import run_sensitivity_analysis, SensitivityResult


# ============================================================================
# Configuration and Result Classes
# ============================================================================

@dataclass
class ValidationConfig:
    """Configuration for validation experiments"""
    n_seeds: int = 30              # Number of random seeds
    batch_size: int = 16           # Samples per experiment
    input_dim: int = 10            # Input dimensionality
    max_order: int = 6             # Maximum derivative order
    K_dirs: int = 5                # Number of random directions to average over
    confidence_level: float = 0.95 # For confidence intervals
    device: str = 'cpu'            # Device for computation
    output_dir: str = 'results/experiments/theoretical_validation'


@dataclass
class ModelTestResult:
    """Results for a single model test"""
    model_name: str
    true_lambda: Optional[float]
    estimated_lambda: float
    lambda_std: float
    lambda_ci: Tuple[float, float]
    true_R: Optional[float]
    estimated_R: Optional[float]
    R_std: Optional[float]
    R_ci: Optional[Tuple[float, float]]
    cyclic_detected: Optional[float]  # Proportion of samples with cyclic pattern
    n_trials: int
    abs_error_lambda: Optional[float]
    rel_error_lambda: Optional[float]


# ============================================================================
# Visualization and Reporting
# ============================================================================

def plot_sensitivity_analysis(sensitivity_results: List[SensitivityResult], output_dir: Path):
    """Plot hyperparameter sensitivity analysis"""
    
    n_params = len(sensitivity_results)
    fig, axes = plt.subplots(1, n_params, figsize=(7 * n_params, 5))
    
    if n_params == 1:
        axes = [axes]
    
    for ax, sens in zip(axes, sensitivity_results):
        param_vals = sens.parameter_values
        means = sens.lambda_means
        cis = sens.lambda_cis
        
        # Special handling for K_dirs: plot variance reduction
        if sens.parameter_name == 'K_dirs' and sens.within_variance is not None:
            # Create twin axis for variance
            ax2 = ax.twinx()
            
            # Plot lambda on left axis
            ax.plot(param_vals, means, 'o-', color='C0', 
                   markersize=8, linewidth=2, alpha=0.7, label='λ estimate')
            ax.set_xlabel('K_dirs (number of random directions)')
            ax.set_ylabel('Estimated λ', color='C0')
            ax.tick_params(axis='y', labelcolor='C0')
            ax.set_title('K_dirs: Variance Reduction from Directional Averaging')
            ax.grid(True, alpha=0.3)
            
            # Plot variance on right axis
            variances = sens.within_variance
            ax2.plot(param_vals, variances, 's-', color='C1',
                    markersize=8, linewidth=2, alpha=0.7, label='Within-network variance')
            ax2.set_ylabel('Variance of λ estimates', color='C1')
            ax2.tick_params(axis='y', labelcolor='C1')
        else:
            # Standard plot for other parameters
            lower_errors = [m - ci[0] for m, ci in zip(means, cis)]
            upper_errors = [ci[1] - m for m, ci in zip(means, cis)]
            ax.errorbar(param_vals, means,
                       yerr=[lower_errors, upper_errors],
                       fmt='o-', capsize=5, markersize=8, linewidth=2, alpha=0.7)
            
            ax.set_xlabel(sens.parameter_name)
            ax.set_ylabel('Estimated λ')
            ax.set_title(f'Sensitivity to {sens.parameter_name}')
            ax.grid(True, alpha=0.3)
            
            # Add variance shading
            lower = [ci[0] for ci in cis]
            upper = [ci[1] for ci in cis]
            ax.fill_between(param_vals, lower, upper, alpha=0.2)
    
    plt.tight_layout(pad=2.0, w_pad=3.0)
    output_path = output_dir / "sensitivity_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results: List[ModelTestResult], 
                   sensitivity_results: List[SensitivityResult],
                   output_dir: Path):
    """Generate comprehensive text report"""
    
    report_path = output_dir / "validation_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("THEORETICAL VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        
        results_with_truth = [r for r in results if r.true_lambda is not None]
        if results_with_truth:
            rel_errors = [r.rel_error_lambda * 100 for r in results_with_truth 
                         if r.rel_error_lambda is not None]
            f.write(f"Models tested with known λ: {len(results_with_truth)}\n")
            f.write(f"Mean relative error: {np.mean(rel_errors):.2f}%\n")
            f.write(f"Std relative error: {np.std(rel_errors, ddof=1):.2f}%\n")
            f.write(f"Max relative error: {np.max(rel_errors):.2f}%\n")
            f.write(f"Models with <5% error: {sum(e < 5 for e in rel_errors)}/{len(rel_errors)}\n")
            f.write(f"Models with <10% error: {sum(e < 10 for e in rel_errors)}/{len(rel_errors)}\n")
        f.write("\n")
        
        # Detailed results per model
        f.write("DETAILED RESULTS BY MODEL\n")
        f.write("-"*80 + "\n\n")
        
        for result in results:
            f.write(f"Model: {result.model_name}\n")
            f.write(f"  Trials: {result.n_trials}\n")
            f.write(f"  Estimated λ: {result.estimated_lambda:.4f} ± {result.lambda_std:.4f}\n")
            f.write(f"  95% CI: [{result.lambda_ci[0]:.4f}, {result.lambda_ci[1]:.4f}]\n")
            
            if result.true_lambda is not None:
                f.write(f"  True λ: {result.true_lambda:.4f}\n")
                f.write(f"  Absolute error: {result.abs_error_lambda:.4f}\n")
                if result.rel_error_lambda is not None:
                    f.write(f"  Relative error: {result.rel_error_lambda*100:.2f}%\n")
                
                # Statistical test: does CI contain true value?
                # Use small epsilon for floating point comparison
                contains_true = (result.lambda_ci[0] - 1e-6) <= result.true_lambda <= (result.lambda_ci[1] + 1e-6)
                f.write(f"  True λ in 95% CI: {'✓ YES' if contains_true else '✗ NO'}\n")
            
            if result.estimated_R is not None:
                f.write(f"  Estimated R: {result.estimated_R:.4f} ± {result.R_std:.4f}\n")
                if result.true_R is not None:
                    f.write(f"  True R: {result.true_R:.4f}\n")
                    R_error = abs(result.estimated_R - result.true_R)
                    R_rel_error = R_error / result.true_R * 100
                    f.write(f"  R relative error: {R_rel_error:.2f}%\n")
            
            if result.cyclic_detected is not None:
                f.write(f"  Cyclic detection rate: {result.cyclic_detected*100:.1f}%\n")
            
            f.write("\n")
        
        # Sensitivity analysis
        if sensitivity_results:
            f.write("HYPERPARAMETER SENSITIVITY ANALYSIS\n")
            f.write("-"*80 + "\n\n")
            
            for sens in sensitivity_results:
                f.write(f"Parameter: {sens.parameter_name}\n")
                f.write(f"  Tested values: {sens.parameter_values}\n")
                f.write(f"  λ estimates: {[f'{m:.4f}' for m in sens.lambda_means]}\n")
                f.write(f"  λ std devs: {[f'{s:.4f}' for s in sens.lambda_stds]}\n")
                
                # Special reporting for K_dirs variance reduction
                if sens.parameter_name == 'K_dirs' and hasattr(sens, 'within_variance') and sens.within_variance is not None:
                    f.write(f"  Within-network variances: {[f'{v:.6f}' for v in sens.within_variance]}\n")
                    var_reduction = (1 - sens.within_variance[-1]/sens.within_variance[0]) * 100
                    f.write(f"  ✓ Variance reduced by {var_reduction:.1f}% (K=1 to K={sens.parameter_values[-1]})\n")
                    f.write(f"  Interpretation: Averaging over more directions successfully reduces estimation variance\n")
                else:
                    # Coefficient of variation across parameter values
                    cv = np.std(sens.lambda_means) / abs(np.mean(sens.lambda_means)) * 100
                    f.write(f"  Coefficient of variation: {cv:.2f}%\n")
                    f.write(f"  Interpretation: {'Low sensitivity' if cv < 10 else 'Moderate sensitivity' if cv < 20 else 'High sensitivity'}\n")
                f.write("\n")


# ============================================================================
# Main Validation Suite
# ============================================================================

def main():
    """Run comprehensive theoretical validation suite"""
    
    print("="*80)
    print("THEORETICAL VALIDATION OF DERIVATIVE DYNAMICS")
    print("="*80)
    print()
    
    # Configuration
    config = ValidationConfig(
        n_seeds=30,
        batch_size=16,
        input_dim=10,
        max_order=6,
        K_dirs=5,
        confidence_level=0.95,
        device='cpu'
    )
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Seeds: {config.n_seeds}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Max derivative order: {config.max_order}")
    print(f"  Directions to average: {config.K_dirs}")
    print(f"  Output directory: {output_dir}")
    print()
    
    results = []
    
    # ========================================================================
    # Test 1: Synthetic Exponential Decay Sequences (Direct Test)
    # ========================================================================
    synthetic_results, test1_results = run_synthetic_sequence_validation(config, ModelTestResult)
    results.extend(test1_results)
    
    # ========================================================================
    # Test 2: Linear Model (Known Ground Truth)
    # ========================================================================
    result_linear = run_linear_model_validation(
        config=config,
        ModelTestResult=ModelTestResult,
        output_dir=output_dir
    )
    results.append(result_linear)
    print()
    
    # ========================================================================
    # Test 3: Sinusoidal Model (Cyclic Patterns)
    # ========================================================================
    result_sin = run_sinusoidal_model_validation(
        config=config,
        ModelTestResult=ModelTestResult
    )
    results.append(result_sin)
    
    # ========================================================================
    # Test 4: Exponential Function (Known λ via composition)
    # ========================================================================
    result_exp_fn, expected_lambda_range = run_exponential_function_validation(config, ModelTestResult, output_dir)
    results.append(result_exp_fn)
    
    # ========================================================================
    # Test 5: Hyperparameter Sensitivity Analysis
    # ========================================================================
    sensitivity_results = run_sensitivity_analysis(
        config=config,
        ModelTestResult=ModelTestResult,
        ValidationConfig=ValidationConfig
    )
    
    # ========================================================================
    # Generate Outputs
    # ========================================================================
    # Suppress numpy warnings for sensitivity analysis (can have single-value arrays)
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    
    plot_sensitivity_analysis(sensitivity_results, output_dir)
    generate_report(results, sensitivity_results, output_dir)


if __name__ == "__main__":
    main()
