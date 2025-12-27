"""
Helper for Test 4: Exponential function validation (f(x) = exp(a·x)).
Validates that loss-derivative lambda is in an expected positive range.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
import os
# Add FlatGrad root to path (go up 3 levels: helpers -> theoretical_validation -> experiments -> FlatGrad)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from flatgrad.derivatives import compute_directional_derivatives, sample_unit_directions


def plot_exponential_function_derivatives(config, output_dir: Path):
    """
    Plot derivative magnitudes vs order for exponential function with fitted lambda slope.
    Shows how derivatives grow exponentially with order and visualizes the estimated lambda.
    """
    print("  Generating derivative growth plot for Test 4...")
    
    a = 2.0
    
    class ExpFunctionModel(torch.nn.Module):
        def __init__(self, a: float):
            super().__init__()
            self.a = a
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.exp(self.a * x.sum(dim=1, keepdim=True))
    
    # Create model for visualization
    torch.manual_seed(42)
    np.random.seed(42)
    model_exp = ExpFunctionModel(a).to(config.device)
    model_exp.eval()
    
    # Evaluate derivatives at the origin
    inputs_exp = torch.zeros(config.batch_size, config.input_dim, device=config.device)
    labels_exp = torch.zeros(config.batch_size, device=config.device)
    
    # Return the model output as the "loss" so directional derivatives are taken on f(x)
    def loss_fn_exp(logits, labels, reduction='none'):
        return logits.squeeze(1)
    
    # Use a fixed direction whose components sum to 1 so derivatives align with a^k exactly
    base_direction = torch.ones_like(inputs_exp)
    base_direction = base_direction / inputs_exp.shape[1]  # sum of components = 1
    derivs = compute_directional_derivatives(
        model=model_exp,
        inputs=inputs_exp,
        labels=labels_exp,
        directions=base_direction,
        loss_fn=loss_fn_exp,
        min_order=1,
        max_order=config.max_order,
        create_graph=True
    )
    
    # Get mean magnitude per order
    orders = np.arange(1, config.max_order + 1)
    mean_magnitudes = np.array([d.abs().mean().item() for d in derivs])
    mean_magnitudes = np.maximum(mean_magnitudes, 1e-12)  # floor to avoid log(0)
    
    # Compute log of magnitudes for lambda estimation
    log_magnitudes = np.log(mean_magnitudes)
    
    # Fit line: log(d_n) = log(A) + lambda * n
    weights = 1.0 / (1.0 + 0.1 * orders)
    coeffs = np.polyfit(orders, log_magnitudes, deg=1, w=weights)
    lambda_slope = coeffs[0]
    intercept = coeffs[1]
    fitted_line = lambda_slope * orders + intercept
    
    # Theoretical values: derivatives are exactly a^k
    theoretical_magnitudes = a ** orders
    theoretical_log = np.log(theoretical_magnitudes)
    
    # Create figure with two subplots
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Linear scale - Derivative magnitudes
    ax1.plot(orders, mean_magnitudes, 'o-', markersize=8, linewidth=2, 
             label='Observed derivatives', color='darkorange')
    ax1.plot(orders, theoretical_magnitudes, 's--', markersize=6, linewidth=1.5, 
             label=f'Theoretical (a^n, a={a})', color='gray', alpha=0.7)
    ax1.set_xlabel('Derivative Order (n)', fontsize=12)
    ax1.set_ylabel('Mean |D^n f|', fontsize=12)
    ax1.set_title(f'Exponential Function: Derivative Growth\n(Linear Scale, f(x)=exp({a}·x))', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(orders)
    
    # Add text annotation showing growth behavior
    textstr = f'Expected: D^n f ∝ a^n = {a}^n\nActual: exponential growth with λ ≈ log({a}) = {np.log(a):.3f}'
    ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Log scale - Show lambda as slope
    ax2.plot(orders, log_magnitudes, 'o', markersize=8, 
             label='log(|D^n f|)', color='darkorange')
    ax2.plot(orders, theoretical_log, 's', markersize=6, 
             label=f'Theoretical log(a^n)', color='gray', alpha=0.7)
    ax2.plot(orders, fitted_line, '--', linewidth=2, color='red',
             label=f'Fitted line (slope = λ = {lambda_slope:.4f})')
    ax2.set_xlabel('Derivative Order (n)', fontsize=12)
    ax2.set_ylabel('log(Mean |D^n f|)', fontsize=12)
    ax2.set_title(f'Exponential Function: Lambda Estimation from Slope\n(Log Scale)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(orders)
    
    # Add interpretation text
    expected_lambda = np.log(a)
    error = abs(lambda_slope - expected_lambda)
    if error < 0.01:
        interpretation = f'λ ≈ log({a}) ✓\nExponential growth\nλ = {lambda_slope:.4f}'
        color = 'lightgreen'
    else:
        interpretation = f'λ ≠ log({a}) ✗\nExpected: {expected_lambda:.4f}\nGot: {lambda_slope:.4f}'
        color = 'lightcoral'
    
    ax2.text(0.05, 0.95, interpretation, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "exponential_derivative_growth.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved Test 4 plot to: {output_path}")
    plt.close()


def run_exponential_function_validation(config, ModelTestResult, output_dir: Path):
    """Run the exponential function test and return a ModelTestResult."""
    print("Test 4: Exponential Function (f(x)=exp(a·x))")
    print("-"*80)
    a = 2.0
    expected_lambda_range = (0.5, 1.0)  # log(2) ≈ 0.693
    print(f"  Using a = {a}  -> expected λ ≈ log(a) = {np.log(a):.3f}")
    print(f"  We check loss-derivative λ is in [{expected_lambda_range[0]}, {expected_lambda_range[1]}]")
    
    class ExpFunctionModel(torch.nn.Module):
        def __init__(self, a: float):
            super().__init__()
            self.a = a
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.exp(self.a * x.sum(dim=1, keepdim=True))
    
    torch.manual_seed(42)
    np.random.seed(42)
    model_exp = ExpFunctionModel(a).to(config.device)
    model_exp.eval()
    # Evaluate derivatives at the origin to match the analytic slope λ = log(a)
    inputs_exp = torch.zeros(config.batch_size, config.input_dim, device=config.device)
    # Dummy labels; we will differentiate the model output directly (not the MSE loss)
    labels_exp = torch.zeros(config.batch_size, device=config.device)
    
    # Return the model output as the "loss" so directional derivatives are taken on f(x)
    def loss_fn_exp(logits, labels, reduction='none'):
        return logits.squeeze(1)
    
    # Use a fixed direction whose components sum to 1 so derivatives align with a^k exactly
    base_direction = torch.ones_like(inputs_exp)
    base_direction = base_direction / inputs_exp.shape[1]  # sum of components = 1
    derivs = compute_directional_derivatives(
        model=model_exp,
        inputs=inputs_exp,
        labels=labels_exp,
        directions=base_direction,
        loss_fn=loss_fn_exp,
        min_order=1,
        max_order=config.max_order,
        create_graph=True
    )
    derivs_exp = [d.detach() for d in derivs]
    
    # Estimate lambda from derivative magnitudes (log-linear fit)
    orders = np.arange(1, config.max_order + 1)
    mean_mags_exp = np.array([d.abs().mean().item() for d in derivs_exp])
    mean_mags_exp = np.maximum(mean_mags_exp, 1e-12)  # floor to avoid log(0)
    log_mags_exp = np.log(mean_mags_exp)
    weights = 1.0 / (1.0 + 0.1 * orders)
    slope, intercept = np.polyfit(orders, log_mags_exp, deg=1, w=weights)
    lambda_est_exp = slope
    
    print(f"  Estimated λ (slope) = {lambda_est_exp:.4f}")
    print(f"  Derivative magnitudes: {[f'{m:.2e}' for m in mean_mags_exp]}")
    
    in_range = expected_lambda_range[0] <= lambda_est_exp <= expected_lambda_range[1]
    status = "✓ PASS" if in_range else "✗ FAIL"
    print(f"  {status}: λ in expected range [{expected_lambda_range[0]}, {expected_lambda_range[1]}]")
    
    # Generate visualization for Test 4
    plot_exponential_function_derivatives(config, output_dir)
    print()
    
    result_exp_fn = ModelTestResult(
        model_name='exp_function',
        true_lambda=None,
        estimated_lambda=lambda_est_exp,
        lambda_std=0.0,
        lambda_ci=(lambda_est_exp, lambda_est_exp),
        true_R=None,
        estimated_R=None,
        R_std=None,
        R_ci=None,
        cyclic_detected=None,
        n_trials=1,
        abs_error_lambda=None,
        rel_error_lambda=None
    )
    
    return result_exp_fn, expected_lambda_range
