import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flatgrad.derivatives import sample_unit_directions, compute_directional_derivatives
from flatgrad.sampling.lambda_estimation import compute_derivative_ratios, estimate_lambda_from_derivatives
from configs.test_config import TEST_LAMBDA_ESTIMATION_OUTPUT_DIR as OUTPUT_DIR


def test_and_plot_derivative_ratios():
    """
    Test and visualize compute_derivative_ratios function.
    Shows that ratios are computed correctly from consecutive derivatives.
    """
    print("Testing compute_derivative_ratios...")
    device = torch.device('cpu')
    
    # Create synthetic derivatives for testing
    batch_size = 10
    n_orders = 10
    
    # Create derivatives that follow an exponential decay pattern with realistic noise
    # This simulates what we'd see in practice: exponential decay but with stochasticity
    derivatives = []
    decay_factor = 0.5  # Each order is roughly half the previous - this creates d_n ≈ base * (0.5)^(n-1)
    noise_level = 0.1  # 10% multiplicative noise for realism
    
    # Generate base values ONCE for all orders (each sample gets its own base)
    base = torch.rand(batch_size, device=device) * 0.1 + 0.05  # Random base values
    
    for order in range(1, n_orders + 1):
        # Simulate exponential decay with multiplicative noise: d_n = base * (decay_factor)^(n-1) * (1 + noise)
        # The noise is different for each sample and each order, simulating real-world variance
        clean_value = base * (decay_factor ** (order - 1))
        
        # Add multiplicative Gaussian noise (centered at 1, scaled by noise_level)
        # This preserves the sign and overall magnitude while adding realistic variation
        noise = 1.0 + torch.randn(batch_size, device=device) * noise_level
        d_n = clean_value * noise
        
        # Ensure values stay positive (since we use absolute values in lambda estimation anyway)
        d_n = torch.abs(d_n)
        
        derivatives.append(d_n)
    
    # Compute ratios
    ratios = compute_derivative_ratios(derivatives)  # [batch_size, n_ratios]
    ratios_np = ratios.detach().cpu().numpy()
    derivatives_np = [d.detach().cpu().numpy() for d in derivatives]
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Derivative values across orders
    ax1 = axes[0]
    orders = list(range(1, n_orders + 1))
    for i in range(min(5, batch_size)):  # Show first 5 samples
        deriv_values = [d[i] for d in derivatives_np]
        ax1.plot(orders, deriv_values, marker='o', label=f'Sample {i+1}', alpha=0.7)
    ax1.set_xlabel('Derivative Order')
    ax1.set_ylabel('Derivative Value')
    ax1.set_title('Derivative Values by Order\n(First 5 samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Ratios between consecutive derivatives (averaged across samples)
    ax2 = axes[1]
    n_ratios = ratios.shape[1]
    ratio_orders = list(range(1, n_ratios + 1))  # Ratio 1 = d2/d1, ratio 2 = d3/d2, etc.
    
    # Compute mean and std across samples for each ratio index
    mean_ratios = ratios_np.mean(axis=0)  # Average across batch dimension
    std_ratios = ratios_np.std(axis=0)    # Std across batch dimension
    
    # Plot average with error bars
    ax2.errorbar(ratio_orders, mean_ratios, yerr=std_ratios, 
                 marker='o', capsize=5, capthick=2, linewidth=2,
                 label='Mean ± Std across samples', color='blue')
    
    # Show expected ratio
    # We artificially created derivatives with decay_factor = 0.5
    # This means d_n = base * (0.5)^(n-1), so d_{n+1}/d_n = (base * 0.5^n) / (base * 0.5^(n-1)) = 0.5
    # The ratio plot verifies our compute_derivative_ratios function works correctly
    ax2.axhline(y=decay_factor, color='r', linestyle='--', linewidth=2, 
                label=f'Expected ({decay_factor}) - synthetic decay factor')
    ax2.set_xlabel('Ratio Index (d_{n+1}/d_n)')
    ax2.set_ylabel('Ratio Value')
    ax2.set_title('Derivative Ratios (Averaged Across Samples)\n(d_{n+1}/d_n for consecutive orders)\nMean should ≈ 0.5 if computation is correct')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "derivative_ratios.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"  Computed ratios shape: {ratios.shape}")
    print(f"  Mean ratio (should be ~{decay_factor:.2f} for synthetic data with decay_factor={decay_factor}): {ratios_np.mean():.4f}")
    print(f"  Std ratio: {ratios_np.std():.4f}\n")


def compute_lambda_per_sample(derivatives, abs_derivatives=True, min_first_derivative=1e-8):
    """
    Helper function to compute lambda for each sample individually.
    
    Returns:
        List of lambda values, one per sample (or None for invalid samples)
    """
    if len(derivatives) < 2:
        return [None] * derivatives[0].shape[0] if len(derivatives) > 0 else []
    
    if abs_derivatives:
        derivatives = [d.abs() for d in derivatives]
    
    try:
        ratios = compute_derivative_ratios(derivatives)  # [B, n_ratios]
    except ValueError:
        return [None] * derivatives[0].shape[0]
    
    batch_size = ratios.shape[0]
    ratios_cpu = ratios.detach().cpu()
    first_derivatives_cpu = derivatives[0].detach().cpu()
    
    lambda_per_sample = []
    for sample_idx in range(batch_size):
        first_deriv = first_derivatives_cpu[sample_idx].item()
        if first_deriv < min_first_derivative:
            lambda_per_sample.append(None)
            continue
        
        sample_ratios = ratios_cpu[sample_idx].numpy()
        log_ratios = np.log(sample_ratios + 1e-12)
        lambda_val = np.mean(log_ratios)
        
        if np.isfinite(lambda_val):
            lambda_per_sample.append(lambda_val)
        else:
            lambda_per_sample.append(None)
    
    return lambda_per_sample


def test_and_plot_lambda_estimation():
    """
    Test and visualize estimate_lambda_from_derivatives function.
    Shows derivatives across orders for different directions, and computed lambda values.
    """
    print("Testing estimate_lambda_from_derivatives...")
    device = torch.device('cpu')
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    ).to(device)
    
    # Use multiple samples (each will have a different random direction)
    batch_size = 8
    inputs = torch.randn(batch_size, 10, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    
    # Sample directions (one per sample)
    directions = sample_unit_directions(batch_size=batch_size, input_shape=(10,), device=device)
    
    # Compute derivatives
    def loss_fn(logits, labels, reduction='none'):
        return F.cross_entropy(logits, labels, reduction=reduction)
    
    max_order = 4
    derivatives = compute_directional_derivatives(
        model=model,
        inputs=inputs,
        labels=labels,
        directions=directions,
        loss_fn=loss_fn,
        min_order=1,
        max_order=max_order,
        create_graph=True  # Need True for higher-order derivatives
    )
    
    # Compute lambda per sample for visualization
    lambda_per_sample = compute_lambda_per_sample(derivatives, abs_derivatives=True)
    valid_lambdas = [l for l in lambda_per_sample if l is not None]
    
    # Compute overall lambda (averaged)
    overall_lambda = estimate_lambda_from_derivatives(derivatives, abs_derivatives=True)
    
    # Convert to numpy for plotting
    derivatives_np = [d.detach().cpu().numpy() for d in derivatives]
    orders = list(range(1, max_order + 1))
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Derivatives across orders for each direction/sample
    ax1 = axes[0]
    for i in range(batch_size):
        if lambda_per_sample[i] is not None:
            deriv_values = [d[i] for d in derivatives_np]
            ax1.plot(orders, deriv_values, marker='o', label=f'Dir {i+1} (λ={lambda_per_sample[i]:.3f})', 
                    alpha=0.7, linewidth=2)
    
    ax1.set_xlabel('Derivative Order')
    ax1.set_ylabel('|Derivative Value|')
    ax1.set_title(f'Directional Derivatives by Order\n(Each line = one direction, λ shown in legend)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Lambda values for each direction, plus overall average
    ax2 = axes[1]
    valid_indices = [i for i, l in enumerate(lambda_per_sample) if l is not None]
    lambda_values_plot = [lambda_per_sample[i] for i in valid_indices]
    
    # Bar chart of individual lambdas
    bars = ax2.bar(range(len(valid_indices)), lambda_values_plot, alpha=0.7, 
                   edgecolor='black', label='Per-direction λ')
    
    # Add horizontal line for overall average
    if overall_lambda is not None:
        ax2.axhline(y=overall_lambda, color='r', linestyle='--', linewidth=2, 
                   label=f'Average λ = {overall_lambda:.3f}')
    
    ax2.set_xlabel('Direction Index')
    ax2.set_ylabel('λ Value')
    ax2.set_title('Lambda Values by Direction\n(Red dashed line = overall average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(len(valid_indices)))
    ax2.set_xticklabels([f'Dir {i+1}' for i in valid_indices])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "lambda_estimation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"  Computed {len(valid_lambdas)} valid lambda values")
    if valid_lambdas:
        print(f"  Lambda range: [{min(valid_lambdas):.4f}, {max(valid_lambdas):.4f}]")
        print(f"  Mean lambda: {np.mean(valid_lambdas):.4f}")
    if overall_lambda is not None:
        print(f"  Overall averaged lambda: {overall_lambda:.4f}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Lambda Estimation Functions")
    print("=" * 60)
    print()
    
    test_and_plot_derivative_ratios()
    test_and_plot_lambda_estimation()
