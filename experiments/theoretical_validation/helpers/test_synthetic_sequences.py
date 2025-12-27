"""
Test 1: Synthetic Exponential Decay Sequences

Validates lambda estimation against synthetic derivative sequences with known ground truth.
Tests multiple decay factors to ensure robustness across different lambda values.
"""

import numpy as np
import torch
from typing import List, Tuple
from scipy import stats

import sys
import os
# Add FlatGrad root to path (go up 3 levels: helpers -> theoretical_validation -> experiments -> FlatGrad)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from flatgrad.sampling.lambda_estimation import estimate_lambda_from_derivatives


def run_synthetic_sequence_validation(
    config,
    ModelTestResult
) -> Tuple[List, List]:
    """
    Run synthetic exponential decay sequence validation.
    
    Args:
        config: ValidationConfig instance
        ModelTestResult: ModelTestResult dataclass type
    
    Returns:
        Tuple of (results_list, synthetic_results_list)
    """
    print("Test 1: Synthetic Exponential Decay Sequences (Direct Test)")
    print("-"*80)
    print("  Testing multiple decay rates with known ground truth λ values")
    print()
    
    synthetic_results = []
    
    # Test multiple decay factors to validate across different λ values
    decay_factors_to_test = [
        (0.1, "Very fast decay"),
        (0.3, "Fast decay"),
        (0.5, "Moderate decay"),
        (0.7, "Slow decay"),
        (0.9, "Very slow decay"),
        (1.0, "No decay (constant)"),
        (1.2, "Growth")
    ]
    
    for decay_factor, description in decay_factors_to_test:
        true_lambda_exp = np.log(decay_factor)
        
        print(f"  [{description}] decay_factor = {decay_factor}")
        print(f"    True λ = log({decay_factor}) = {true_lambda_exp:.4f}")
        
        # Generate synthetic derivatives with exact exponential decay
        # d_n = A * (decay_factor)^n
        lambda_estimates = []
        
        for seed in range(config.n_seeds):
            torch.manual_seed(42 + seed + int(decay_factor * 1000))  # Unique seed per factor
            np.random.seed(42 + seed + int(decay_factor * 1000))
            
            # Create synthetic derivatives
            batch_size = config.batch_size
            A = torch.rand(batch_size) * 2.0 + 0.5  # Amplitude between 0.5 and 2.5
            
            derivatives = []
            for n in range(1, config.max_order + 1):
                d_n = A * (decay_factor ** n)
                derivatives.append(d_n)
            
            # Estimate lambda from these synthetic derivatives
            lambda_est = estimate_lambda_from_derivatives(derivatives, abs_derivatives=True)
            if lambda_est is not None:
                lambda_estimates.append(lambda_est)
        
        # Aggregate statistics for this decay factor
        lambda_estimates = np.array(lambda_estimates)
        lambda_mean = np.mean(lambda_estimates)
        lambda_std = np.std(lambda_estimates, ddof=1)
        sem_lambda = lambda_std / np.sqrt(len(lambda_estimates))
        t_crit = stats.t.ppf((1 + config.confidence_level) / 2, len(lambda_estimates) - 1)
        lambda_ci = (lambda_mean - t_crit * sem_lambda, lambda_mean + t_crit * sem_lambda)
        
        abs_error = abs(lambda_mean - true_lambda_exp)
        rel_error = abs_error / abs(true_lambda_exp) if true_lambda_exp != 0 else abs_error
        
        result_synthetic = ModelTestResult(
            model_name=f'synthetic_r{decay_factor:.1f}',
            true_lambda=true_lambda_exp,
            estimated_lambda=lambda_mean,
            lambda_std=lambda_std,
            lambda_ci=lambda_ci,
            true_R=None,
            estimated_R=None,
            R_std=None,
            R_ci=None,
            cyclic_detected=None,
            n_trials=len(lambda_estimates),
            abs_error_lambda=abs_error,
            rel_error_lambda=rel_error
        )
        synthetic_results.append(result_synthetic)
        
        # Report results for this decay factor
        print(f"    Estimated λ = {lambda_mean:.4f} ± {lambda_std:.4f}")
        print(f"    95% CI: [{lambda_ci[0]:.4f}, {lambda_ci[1]:.4f}]")
        print(f"    Absolute error: {abs_error:.4f}")
        if true_lambda_exp != 0:
            print(f"    Relative error: {rel_error*100:.2f}%")
        
        # Check if CI contains true value
        contains_true = (lambda_ci[0] - 1e-6) <= true_lambda_exp <= (lambda_ci[1] + 1e-6)
        status = "✓ YES" if contains_true else "✗ NO"
        print(f"    True λ in 95% CI: {status}")
        
        # Pass/fail for this decay rate
        if true_lambda_exp != 0:
            passed = rel_error < 0.05 and contains_true  # < 5% error and CI coverage
        else:
            passed = abs_error < 0.05 and contains_true  # < 0.05 absolute error for λ=0
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    Overall: {status}")
        print()
    
    # Aggregate statistics across all synthetic tests
    print("  " + "="*76)
    print("  AGGREGATE STATISTICS (All Synthetic Tests)")
    print("  " + "="*76)
    all_rel_errors = [r.rel_error_lambda * 100 for r in synthetic_results 
                      if r.rel_error_lambda is not None and r.true_lambda != 0]
    all_abs_errors = [r.abs_error_lambda for r in synthetic_results]
    ci_coverage = sum((r.lambda_ci[0] - 1e-6) <= r.true_lambda <= (r.lambda_ci[1] + 1e-6) 
                      for r in synthetic_results)
    
    print(f"  Tests run: {len(synthetic_results)}")
    print(f"  Mean absolute error: {np.mean(all_abs_errors):.4f}")
    print(f"  Std absolute error: {np.std(all_abs_errors, ddof=1):.4f}")
    if all_rel_errors:
        print(f"  Mean relative error: {np.mean(all_rel_errors):.2f}%")
        print(f"  Max relative error: {np.max(all_rel_errors):.2f}%")
        print(f"  Tests with <5% error: {sum(e < 5 for e in all_rel_errors)}/{len(all_rel_errors)}")
    print(f"  95% CI coverage: {ci_coverage}/{len(synthetic_results)} ({ci_coverage/len(synthetic_results)*100:.0f}%)")
    
    # Overall pass/fail
    if all_rel_errors:
        mean_rel_error = np.mean(all_rel_errors)
        overall_pass = mean_rel_error < 5.0 and ci_coverage == len(synthetic_results)
    else:
        overall_pass = ci_coverage == len(synthetic_results)
    
    status = "✓ PASS" if overall_pass else "✗ FAIL"
    print(f"  Overall Test 1: {status}")
    print()
    
    return synthetic_results, synthetic_results
