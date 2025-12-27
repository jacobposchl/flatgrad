"""
Helper for Test 3: Sinusoidal model validation.
Validates cyclic pattern detection in derivative sequences.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import stats

import sys
import os
# Add FlatGrad root to path (go up 3 levels: helpers -> theoretical_validation -> experiments -> FlatGrad)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from flatgrad.derivatives import sample_unit_directions, compute_directional_derivatives
from flatgrad.sampling.lambda_estimation import estimate_lambda_from_derivatives
from flatgrad.sampling.metrics import compute_analytic_radius, detect_cyclic
from flatgrad.sampling.models import create_test_model


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
    """
    Compute directional derivatives averaged over K_dirs random directions.
    """
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
            print(f"    Warning: Direction {k+1}/{K_dirs} failed with error: {e}")
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


def run_sinusoidal_model_validation(config, ModelTestResult):
    """Run Test 3 (sinusoidal model) and check for cyclic patterns."""
    print("Test 3: Sinusoidal Model (Cyclic Patterns)")
    print("-"*80)
    
    device = torch.device(config.device)
    frequency = 2.0
    
    lambda_estimates = []
    R_estimates = []
    cyclic_detections = []
    
    for seed in range(config.n_seeds):
        torch.manual_seed(42 + seed)
        np.random.seed(42 + seed)
        
        model = create_test_model(
            model_type='sinusoidal',
            input_dim=config.input_dim,
            frequency=frequency
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
        
        # Estimate lambda (use absolute values)
        lambda_est = estimate_lambda_from_derivatives(derivatives, abs_derivatives=True)
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
        
        # Check cyclic patterns
        is_cyclic, _, _ = detect_cyclic(derivatives, threshold=0.5, abs_derivatives=False)
        cyclic_detections.append(np.mean(is_cyclic))
    
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
    
    # Cyclic statistics
    cyclic_prop = np.mean(cyclic_detections)
    
    result_sin = ModelTestResult(
        model_name='sinusoidal',
        true_lambda=None,
        estimated_lambda=lambda_mean,
        lambda_std=lambda_std,
        lambda_ci=lambda_ci,
        true_R=None,
        estimated_R=R_mean,
        R_std=R_std,
        R_ci=R_ci,
        cyclic_detected=cyclic_prop,
        n_trials=config.n_seeds,
        abs_error_lambda=None,
        rel_error_lambda=None
    )
    
    print(f"  Estimated λ = {result_sin.estimated_lambda:.4f} ± {result_sin.lambda_std:.4f}")
    print(f"  Cyclic detection rate: {result_sin.cyclic_detected*100:.1f}%")
    if result_sin.cyclic_detected >= 0.5:
        print(f"  ✓ PASS: Cyclic detection ≥ 50%")
    else:
        print(f"  ✗ FAIL: Cyclic detection < 50%")
    print()
    
    return result_sin
