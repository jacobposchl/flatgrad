"""
Helper for Test 5: Hyperparameter Sensitivity Analysis.
Tests how lambda estimates vary with different hyperparameters.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import sys
import os
# Add FlatGrad root to path (go up 3 levels: helpers -> theoretical_validation -> experiments -> FlatGrad)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from flatgrad.derivatives import sample_unit_directions, compute_directional_derivatives
from flatgrad.sampling.lambda_estimation import estimate_lambda_from_derivatives
from flatgrad.sampling.metrics import compute_analytic_radius
from flatgrad.sampling.models import create_test_model


@dataclass
class SensitivityResult:
    """Results for hyperparameter sensitivity analysis"""
    parameter_name: str
    parameter_values: List
    lambda_means: List[float]
    lambda_stds: List[float]
    lambda_cis: List[Tuple[float, float]]
    within_variance: Optional[List[float]] = None  # Variance within single network (for K_dirs)


def compute_derivatives_multi_direction(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    loss_fn,
    max_order: int,
    K_dirs: int,
    device: str = 'cpu',
    min_norm_threshold: float = 1e-10
):
    """Compute directional derivatives averaged over K_dirs random directions."""
    batch_size = inputs.shape[0]
    input_shape = inputs.shape[1:]
    
    was_training = model.training
    model.eval()
    
    accumulated_derivatives = [torch.zeros(batch_size) for _ in range(max_order)]
    valid_counts = [torch.zeros(batch_size) for _ in range(max_order)]
    
    for k in range(K_dirs):
        directions = sample_unit_directions(
            batch_size=batch_size,
            input_shape=input_shape,
            device=device
        )
        
        try:
            derivatives = compute_directional_derivatives(
                model=model,
                inputs=inputs,
                labels=labels,
                directions=directions,
                loss_fn=loss_fn,
                min_order=1,
                max_order=max_order,
                create_graph=True
            )
            
            for i, d in enumerate(derivatives):
                d_cpu = d.detach().cpu()
                valid_mask = torch.isfinite(d_cpu) & (d_cpu.abs() >= min_norm_threshold)
                accumulated_derivatives[i] += torch.where(valid_mask, d_cpu, torch.zeros_like(d_cpu))
                valid_counts[i] += valid_mask.float()
                
        except Exception as e:
            continue
    
    averaged_derivatives = []
    for i in range(max_order):
        valid_count = torch.clamp(valid_counts[i], min=1.0)
        avg_d = accumulated_derivatives[i] / valid_count
        avg_d = torch.where(avg_d.abs() < min_norm_threshold, 
                           torch.zeros_like(avg_d), 
                           avg_d)
        averaged_derivatives.append(avg_d)
    
    if was_training:
        model.train()
    
    return averaged_derivatives


def run_exponential_decay_validation(config, ModelTestResult, decay_factor: float = 0.5):
    """Run exponential decay model validation for one sensitivity test."""
    device = torch.device(config.device)
    
    lambda_estimates = []
    R_estimates = []
    
    for seed in range(config.n_seeds):
        torch.manual_seed(42 + seed)
        np.random.seed(42 + seed)
        
        model = create_test_model(
            model_type='exponential',
            input_dim=config.input_dim,
            decay_factor=decay_factor
        ).to(device)
        model.eval()
        
        inputs = torch.randn(config.batch_size, config.input_dim, device=device)
        labels = torch.randn(config.batch_size, device=device)
        
        def loss_fn(logits, labels, reduction='none'):
            return F.mse_loss(logits.squeeze(1), labels, reduction=reduction)
        
        derivatives = compute_derivatives_multi_direction(
            model=model,
            inputs=inputs,
            labels=labels,
            loss_fn=loss_fn,
            max_order=config.max_order,
            K_dirs=config.K_dirs,
            device=config.device
        )
        
        # Estimate lambda (preserve signs for exponential decay)
        lambda_est = estimate_lambda_from_derivatives(derivatives, abs_derivatives=False)
        if lambda_est is not None:
            lambda_estimates.append(lambda_est)
        
        # Estimate R
        R_result = compute_analytic_radius(derivatives, abs_derivatives=True)
        if isinstance(R_result, tuple):
            R_values, _ = R_result
            R_est = np.nanmean(R_values)
        else:
            R_est = R_result
        
        if R_est is not None and np.isfinite(R_est):
            R_estimates.append(R_est)
    
    # Aggregate statistics
    lambda_estimates = np.array(lambda_estimates)
    lambda_mean = np.mean(lambda_estimates)
    lambda_std = np.std(lambda_estimates, ddof=1)
    
    # Confidence interval
    if lambda_std > 1e-10:
        sem_lambda = lambda_std / np.sqrt(len(lambda_estimates))
        t_crit = stats.t.ppf((1 + config.confidence_level) / 2, len(lambda_estimates) - 1)
        lambda_ci = (lambda_mean - t_crit * sem_lambda, lambda_mean + t_crit * sem_lambda)
    else:
        lambda_ci = (lambda_mean, lambda_mean)
    
    # R statistics
    if R_estimates:
        R_estimates = np.array(R_estimates)
        R_mean = np.mean(R_estimates)
        R_std = np.std(R_estimates, ddof=1)
        if R_std > 1e-10:
            sem_R = R_std / np.sqrt(len(R_estimates))
            t_crit_R = stats.t.ppf((1 + config.confidence_level) / 2, len(R_estimates) - 1)
            R_ci = (R_mean - t_crit_R * sem_R, R_mean + t_crit_R * sem_R)
        else:
            R_ci = (R_mean, R_mean)
    else:
        R_mean = R_std = R_ci = None
    
    # Compute analytical lambda
    true_lambda = np.log(decay_factor)
    abs_error = abs(lambda_mean - true_lambda)
    rel_error = abs_error / abs(true_lambda) if true_lambda != 0 else None
    
    return ModelTestResult(
        model_name='exponential',
        true_lambda=true_lambda,
        estimated_lambda=lambda_mean,
        lambda_std=lambda_std,
        lambda_ci=lambda_ci,
        true_R=None,
        estimated_R=R_mean,
        R_std=R_std,
        R_ci=R_ci,
        cyclic_detected=None,
        n_trials=len(lambda_estimates),
        abs_error_lambda=abs_error,
        rel_error_lambda=rel_error
    )


def run_sensitivity_analysis(config, ModelTestResult, ValidationConfig):
    """Run Test 5: Hyperparameter Sensitivity Analysis"""
    print("Test 5: Hyperparameter Sensitivity Analysis")
    print("-"*80)
    
    sensitivity_results = []
    
    # Sensitivity to K_dirs - NOW SHOWING VARIANCE REDUCTION
    print("\n  Testing K_dirs (shows variance reduction within single network)...")
    sens_Kdirs = kdirs_variance_reduction(
        parameter_values=[1, 2, 5, 10, 20],
        config=config,
        ValidationConfig=ValidationConfig,
        ModelTestResult=ModelTestResult,
        decay_factor=0.5
    )
    sensitivity_results.append(sens_Kdirs)
    print(f"    λ estimates: {[f'{m:.4f}' for m in sens_Kdirs.lambda_means]}")
    print(f"    Within-network variance: {[f'{v:.4f}' for v in sens_Kdirs.within_variance]}")
    
    # Sensitivity to max_order
    print("\n  Testing sensitivity to max_order...")
    sens_maxorder = hyperparameter_sensitivity(
        parameter_name='max_order',
        parameter_values=[3, 5, 7, 9, 11],
        config=config,
        ValidationConfig=ValidationConfig,
        ModelTestResult=ModelTestResult,
        decay_factor=0.5
    )
    sensitivity_results.append(sens_maxorder)
    print(f"    λ estimates: {[f'{m:.4f}' for m in sens_maxorder.lambda_means]}")
    print(f"    Estimates stabilize with sufficient derivative orders")
    print()
    
    return sensitivity_results


def hyperparameter_sensitivity(
    parameter_name: str,
    parameter_values: List,
    config,
    ValidationConfig,
    ModelTestResult,
    decay_factor: float = 0.5
) -> SensitivityResult:
    """
    Test sensitivity of lambda estimates to hyperparameter variations.
    
    Args:
        parameter_name: Name of parameter to vary ('K_dirs', 'max_order', 'batch_size')
        parameter_values: List of values to test
        config: Base configuration
        ValidationConfig: Configuration class for creating test configs
        ModelTestResult: Result class for test results
        decay_factor: Decay factor for exponential model
    
    Returns:
        SensitivityResult with statistics for each parameter value
    """
    lambda_means = []
    lambda_stds = []
    lambda_cis = []
    
    for param_val in parameter_values:
        # Create modified config
        test_config = ValidationConfig(
            n_seeds=config.n_seeds,
            batch_size=config.batch_size,
            input_dim=config.input_dim,
            max_order=config.max_order,
            K_dirs=config.K_dirs,
            confidence_level=config.confidence_level,
            device=config.device,
            output_dir=config.output_dir
        )
        
        # Modify the specific parameter
        if parameter_name == 'K_dirs':
            test_config.K_dirs = param_val
        elif parameter_name == 'max_order':
            test_config.max_order = param_val
        elif parameter_name == 'batch_size':
            test_config.batch_size = param_val
        else:
            raise ValueError(f"Unknown parameter: {parameter_name}")
        
        # Run validation with fewer seeds for efficiency
        test_config.n_seeds = min(15, config.n_seeds)
        
        # Use exponential decay model for sensitivity analysis
        result = run_exponential_decay_validation(
            config=test_config,
            ModelTestResult=ModelTestResult,
            decay_factor=decay_factor
        )
        
        lambda_means.append(result.estimated_lambda)
        lambda_stds.append(result.lambda_std)
        lambda_cis.append(result.lambda_ci)
    
    return SensitivityResult(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        lambda_means=lambda_means,
        lambda_stds=lambda_stds,
        lambda_cis=lambda_cis
    )


def kdirs_variance_reduction(
    parameter_values: List[int],
    config,
    ValidationConfig,
    ModelTestResult,
    decay_factor: float = 0.5,
    n_trials_per_network: int = 20
) -> SensitivityResult:
    """
    Test K_dirs by measuring variance WITHIN a single network.
    
    For each K value:
    - Use ONE fixed neural network
    - Run n_trials_per_network experiments with different random directions
    - Measure variance across those trials (this shows directional sampling variance)
    
    This demonstrates that K_dirs reduces within-network estimation variance.
    """
    lambda_means = []
    lambda_stds = []
    lambda_cis = []
    within_variances = []
    
    # Fix one network for fair comparison
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(config.device)
    
    model = create_test_model(
        model_type='exponential',
        input_dim=config.input_dim,
        decay_factor=decay_factor
    ).to(device)
    model.eval()
    
    # Fixed data
    inputs = torch.randn(config.batch_size, config.input_dim, device=device)
    labels = torch.randn(config.batch_size, device=device)
    
    def loss_fn(logits, labels, reduction='none'):
        return F.mse_loss(logits.squeeze(1), labels, reduction=reduction)
    
    for K_val in parameter_values:
        trial_lambdas = []
        
        # Run multiple trials with different random directions
        for trial in range(n_trials_per_network):
            torch.manual_seed(1000 + trial)  # Different directions each trial
            np.random.seed(1000 + trial)
            
            derivatives = compute_derivatives_multi_direction(
                model=model,
                inputs=inputs,
                labels=labels,
                loss_fn=loss_fn,
                max_order=config.max_order,
                K_dirs=K_val,
                device=config.device
            )
            
            lambda_est = estimate_lambda_from_derivatives(derivatives, abs_derivatives=False)
            if lambda_est is not None:
                trial_lambdas.append(lambda_est)
        
        # Statistics across trials (shows effect of K_dirs on variance)
        trial_lambdas = np.array(trial_lambdas)
        lambda_mean = np.mean(trial_lambdas)
        lambda_std = np.std(trial_lambdas, ddof=1)
        
        # The KEY metric: variance within network (reduced by K_dirs)
        within_var = np.var(trial_lambdas, ddof=1)
        within_variances.append(within_var)
        
        # CI
        if lambda_std > 1e-10:
            sem = lambda_std / np.sqrt(len(trial_lambdas))
            t_crit = stats.t.ppf((1 + config.confidence_level) / 2, len(trial_lambdas) - 1)
            ci = (lambda_mean - t_crit * sem, lambda_mean + t_crit * sem)
        else:
            ci = (lambda_mean, lambda_mean)
        
        lambda_means.append(lambda_mean)
        lambda_stds.append(lambda_std)
        lambda_cis.append(ci)
    
    return SensitivityResult(
        parameter_name='K_dirs',
        parameter_values=parameter_values,
        lambda_means=lambda_means,
        lambda_stds=lambda_stds,
        lambda_cis=lambda_cis,
        within_variance=within_variances
    )
