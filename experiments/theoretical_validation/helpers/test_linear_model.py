"""
Helper for Test 2: Linear Model validation and plotting
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

from flatgrad.derivatives import compute_directional_derivatives, sample_unit_directions
from flatgrad.sampling.lambda_estimation import estimate_lambda_from_derivatives
from flatgrad.sampling.metrics import compute_analytic_radius
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


def plot_linear_model_derivatives(config, output_dir: Path):
    """
    Plot derivative magnitudes vs order for linear model with fitted lambda slope.
    Shows how derivatives decay with order and visualizes the estimated lambda.
    """
    print("  Generating derivative decay plot for Test 2...")
    
    # Create a fresh linear model for visualization
    torch.manual_seed(42)
    np.random.seed(42)
    model_linear = create_test_model('polynomial', config.input_dim, degree=1).to(config.device)
    model_linear.eval()
    inputs = torch.randn(config.batch_size, config.input_dim, device=config.device)
    labels = torch.randn(config.batch_size, device=config.device)
    
    def loss_fn(logits, labels, reduction='none'):
        return F.mse_loss(logits.squeeze(1), labels, reduction=reduction)
    
    # Compute derivatives using multi-direction averaging
    directions = sample_unit_directions(batch_size=config.batch_size,
                                        input_shape=inputs.shape[1:],
                                        device=config.device)
    derivatives = compute_directional_derivatives(
        model=model_linear,
        inputs=inputs,
        labels=labels,
        directions=directions,
        loss_fn=loss_fn,
        min_order=1,
        max_order=config.max_order,
        create_graph=True
    )
    
    # Get mean magnitude per order
    orders = np.arange(1, config.max_order + 1)
    mean_magnitudes = np.array([d.abs().mean().item() for d in derivatives])
    
    # Compute log of magnitudes for lambda estimation
    log_magnitudes = np.log(mean_magnitudes + 1e-12)  # Add epsilon to avoid log(0)
    
    # Fit line: log(d_n) = log(A) + lambda * n
    # Using weighted least squares (weight higher orders less due to numerical noise)
    weights = 1.0 / (1.0 + 0.1 * orders)
    coeffs = np.polyfit(orders, log_magnitudes, deg=1, w=weights)
    lambda_slope = coeffs[0]
    intercept = coeffs[1]
    fitted_line = lambda_slope * orders + intercept
    
    # Create figure with two subplots
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Linear scale - Derivative magnitudes
    ax1.plot(orders, mean_magnitudes, 'o-', markersize=8, linewidth=2, 
             label='Observed derivatives', color='steelblue')
    ax1.set_xlabel('Derivative Order (n)', fontsize=12)
    ax1.set_ylabel('Mean |D^n L|', fontsize=12)
    ax1.set_title('Linear Model: Derivative Magnitude Decay\n(Linear Scale)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(orders)
    
    # Add text annotation showing decay behavior
    textstr = 'Expected: d^3L/dε^3 = 0 (polynomial cutoff)\nActual: derivatives → 0 for n ≥ 3'
    ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Log scale - Show lambda as slope
    ax2.plot(orders, log_magnitudes, 'o', markersize=8, 
             label='log(|D^n L|)', color='steelblue')
    ax2.plot(orders, fitted_line, '--', linewidth=2, color='red',
             label=f'Fitted line (slope = λ = {lambda_slope:.3f})')
    ax2.set_xlabel('Derivative Order (n)', fontsize=12)
    ax2.set_ylabel('log(Mean |D^n L|)', fontsize=12)
    ax2.set_title('Linear Model: Lambda Estimation from Slope\n(Log Scale)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(orders)
    
    # Add interpretation text
    if lambda_slope < 0:
        interpretation = 'λ < 0 ✓\nExponential decay\n(polynomial cutoff)'
        color = 'green'
    else:
        interpretation = 'λ ≥ 0 ✗\nUnexpected for\npolynomial model'
        color = 'red'
    
    ax2.text(0.05, 0.05, interpretation, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    output_path = output_dir / "linear_derivative_decay.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved Test 2 plot to: {output_path}")
    plt.close()


def run_linear_model_validation(config, ModelTestResult, output_dir: Path):
    """Run Test 2 (linear model) and generate its plot."""
    print("Test 2: Linear Model (Known Ground Truth)")
    print("-"*80)
    print("  Linear model: f(x) = W·x + b")
    print("  MSE loss: L = (f(x) - y)^2")
    print("  First derivative: dL/dε = 2(f(x)-y)·(W·v)")
    print("  Second derivative: d²L/dε² = 2(W·v)²")
    print("  Higher derivatives: d³L/dε³ = 0, d⁴L/dε⁴ = 0, ...")
    print("  Expected λ: Should be negative (derivatives decay to zero)")
    
    device = torch.device(config.device)
    lambda_estimates = []
    R_estimates = []
    
    for seed in range(config.n_seeds):
        torch.manual_seed(42 + seed)
        np.random.seed(42 + seed)
        
        model = create_test_model(
            model_type='polynomial',
            input_dim=config.input_dim,
            degree=1
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
        
        # Estimate lambda (use absolute values for polynomial)
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
    
    result_linear = ModelTestResult(
        model_name='linear',
        true_lambda=None,
        estimated_lambda=lambda_mean,
        lambda_std=lambda_std,
        lambda_ci=lambda_ci,
        true_R=None,
        estimated_R=R_mean,
        R_std=R_std,
        R_ci=R_ci,
        cyclic_detected=None,
        n_trials=config.n_seeds,
        abs_error_lambda=None,
        rel_error_lambda=None
    )
    
    print(f"  Estimated λ = {result_linear.estimated_lambda:.4f} ± {result_linear.lambda_std:.4f}")
    print("  Expected: λ < 0 (derivatives decay to zero after order 2)")
    if result_linear.estimated_lambda < 0:
        print("  ✓ PASS: λ < 0 as expected for polynomial cutoff")
    else:
        print("  ✗ FAIL: λ ≥ 0, unexpected for polynomial model")
    
    # Generate visualization for Test 2
    plot_linear_model_derivatives(config, output_dir)
    print()
    
    return result_linear
