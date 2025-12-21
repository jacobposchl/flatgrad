import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flatgrad.derivatives import sample_unit_directions, compute_directional_derivatives
from flatgrad.sampling.metrics import (
    WLSResult,
    wls_linear,
    wls_linear_log,
    compute_analytic_radius,
    compute_spectral_edge,
    detect_cyclic
)
from configs.test_config import TEST_METRICS_OUTPUT_DIR as OUTPUT_DIR


def test_wls_linear():
    """
    Test weighted least squares linear fitting.
    """
    print("Testing wls_linear...")
    
    # Create synthetic linear data with noise
    np.random.seed(42)
    n_points = 10
    x = np.linspace(1, 10, n_points)
    true_slope = 2.5
    true_intercept = 1.0
    noise = np.random.normal(0, 0.5, n_points)
    y = true_slope * x + true_intercept + noise
    
    # Test with uniform weights
    result_uniform = wls_linear(x, y)
    
    # Test with custom weights (higher weight on later points)
    weights = np.linspace(0.5, 2.0, n_points)
    result_weighted = wls_linear(x, y, weights=weights)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Uniform weights
    ax1 = axes[0]
    ax1.scatter(x, y, alpha=0.7, label='Data', s=50)
    y_fit_uniform = result_uniform.slope * x + result_uniform.intercept
    ax1.plot(x, y_fit_uniform, 'r-', linewidth=2, label=f'Fit (slope={result_uniform.slope:.3f}±{result_uniform.slope_std:.3f})')
    ax1.plot(x, true_slope * x + true_intercept, 'g--', alpha=0.5, label=f'True (slope={true_slope:.1f})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('WLS Linear Fit (Uniform Weights)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weighted
    ax2 = axes[1]
    ax2.scatter(x, y, alpha=0.7, label='Data', s=50)
    y_fit_weighted = result_weighted.slope * x + result_weighted.intercept
    ax2.plot(x, y_fit_weighted, 'r-', linewidth=2, label=f'Fit (slope={result_weighted.slope:.3f}±{result_weighted.slope_std:.3f})')
    ax2.plot(x, true_slope * x + true_intercept, 'g--', alpha=0.5, label=f'True (slope={true_slope:.1f})')
    # Show weights as size
    sizes = weights * 50
    ax2.scatter(x, y, s=sizes, alpha=0.3, c='blue', label='Weight size')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('WLS Linear Fit (Weighted)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "wls_linear.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"\n  === WLS Linear Results (Uniform Weights) ===")
    print(f"  Fitted slope: {result_uniform.slope:.4f} ± {result_uniform.slope_std:.4f} (true: {true_slope:.1f})")
    print(f"  Fitted intercept: {result_uniform.intercept:.4f} ± {result_uniform.intercept_std:.4f} (true: {true_intercept:.1f})")
    print(f"  Slope 95% CI: [{result_uniform.slope_ci_lower:.4f}, {result_uniform.slope_ci_upper:.4f}]")
    print(f"  R-squared: {result_uniform.r_squared:.4f}")
    print(f"  p-value: {result_uniform.p_value:.6f}")
    print(f"  Degrees of freedom: {result_uniform.degrees_of_freedom}")
    
    print(f"\n  === WLS Linear Results (Weighted) ===")
    print(f"  Fitted slope: {result_weighted.slope:.4f} ± {result_weighted.slope_std:.4f} (true: {true_slope:.1f})")
    print(f"  R-squared: {result_weighted.r_squared:.4f}")
    print()


def test_wls_linear_log():
    """
    Test weighted least squares with logarithmic transformation.
    """
    print("Testing wls_linear_log...")
    
    # Create synthetic exponential decay data
    np.random.seed(42)
    n_points = 10
    x = np.linspace(1, 10, n_points)
    true_decay_rate = -0.3  # Slope in log space
    true_intercept = 2.0
    noise = np.random.normal(0, 0.1, n_points)
    y = true_decay_rate * np.log(x + 1e-12) + true_intercept + noise
    
    result = wls_linear_log(x, y)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Original scale
    ax1 = axes[0]
    ax1.scatter(x, y, alpha=0.7, label='Data', s=50)
    y_fit = result.slope * np.log(x + 1e-12) + result.intercept
    ax1.plot(x, y_fit, 'r-', linewidth=2, label=f'Fit (decay={result.slope:.3f})')
    y_true = true_decay_rate * np.log(x + 1e-12) + true_intercept
    ax1.plot(x, y_true, 'g--', alpha=0.5, label=f'True (decay={true_decay_rate:.1f})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('WLS Linear Log Fit (Original Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    ax2 = axes[1]
    x_log = np.log(x + 1e-12)
    ax2.scatter(x_log, y, alpha=0.7, label='Data', s=50)
    ax2.plot(x_log, y_fit, 'r-', linewidth=2, label='Fit')
    ax2.plot(x_log, y_true, 'g--', alpha=0.5, label='True')
    ax2.set_xlabel('log(x)')
    ax2.set_ylabel('y')
    ax2.set_title('WLS Linear Log Fit (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "wls_linear_log.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"\n  === WLS Linear Log Results ===")
    print(f"  Fitted decay rate (slope): {result.slope:.4f} ± {result.slope_std:.4f} (true: {true_decay_rate:.1f})")
    print(f"  Fitted intercept: {result.intercept:.4f} ± {result.intercept_std:.4f} (true: {true_intercept:.1f})")
    print(f"  R-squared: {result.r_squared:.4f}")
    print(f"  p-value: {result.p_value:.6f}")
    print()


def test_compute_analytic_radius():
    """
    Test analytic radius computation from factorial-normalized derivatives.
    """
    print("Testing compute_analytic_radius...")
    device = torch.device('cpu')
    
    # Create synthetic derivatives that follow d_n ~ R^(-n) * n!
    # For testing, use R = 2.0
    true_R = 2.0
    batch_size = 5
    n_orders = 8
    
    # Generate derivatives: d_n = R^(-n) * n! * base
    derivatives = []
    for order in range(1, n_orders + 1):
        factorial = math.factorial(order)
        base = torch.rand(batch_size, device=device) * 0.1 + 0.05
        d_n = base * (true_R ** (-order)) * factorial
        # Add small noise
        noise = torch.randn(batch_size, device=device) * 0.01
        derivatives.append(d_n + noise)
    
    # Compute analytic radius
    R_result = compute_analytic_radius(derivatives, abs_derivatives=True)
    R_values, R_conf = R_result
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Factorial-normalized coefficients
    ax1 = axes[0]
    orders = np.arange(1, n_orders + 1)
    factorials = np.array([math.factorial(n) for n in orders])
    
    for sample_idx in range(min(3, batch_size)):
        deriv_seq = np.array([d[sample_idx].item() for d in derivatives])
        a_n = deriv_seq / factorials
        ax1.plot(orders, a_n, marker='o', label=f'Sample {sample_idx+1}', alpha=0.7)
        # Show expected pattern: a_n ~ R^(-n)
        expected_a_n = deriv_seq[0].item() / factorials[0] * (true_R ** (-orders))
        ax1.plot(orders, expected_a_n, '--', alpha=0.3)
    
    ax1.set_xlabel('Derivative Order n')
    ax1.set_ylabel('Factorial-Normalized Coefficient a_n = |d_n| / n!')
    ax1.set_title('Factorial-Normalized Coefficients\n(Should show exponential decay ~ R^(-n))')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Estimated R values
    ax2 = axes[1]
    valid_mask = np.isfinite(R_values)
    valid_R = R_values[valid_mask]
    sample_indices = np.arange(batch_size)[valid_mask]
    
    ax2.bar(sample_indices, valid_R, alpha=0.7, edgecolor='black', label='Estimated R')
    ax2.axhline(y=true_R, color='r', linestyle='--', linewidth=2, label=f'True R = {true_R:.2f}')
    
    # Add confidence intervals if available
    if R_conf is not None and len(valid_R) > 0:
        for i, idx in enumerate(sample_indices):
            conf_item = R_conf[idx]
            # Check if it's a tuple/array with 2 elements (confidence interval)
            if isinstance(conf_item, (tuple, list, np.ndarray)) and len(conf_item) == 2:
                ci_lower, ci_upper = conf_item
                if np.isfinite(ci_lower) and np.isfinite(ci_upper):
                    ax2.plot([idx, idx], [ci_lower, ci_upper], 'k-', linewidth=2, alpha=0.5)
    
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Analytic Radius R')
    ax2.set_title('Estimated Analytic Radius\n(with 95% confidence intervals)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "analytic_radius.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"\n  === Analytic Radius Results ===")
    print(f"  True R: {true_R:.4f}")
    if isinstance(R_result, tuple):
        print(f"  Estimated R values (batch of {batch_size}):")
        for i, R_val in enumerate(R_values):
            if np.isfinite(R_val):
                print(f"    Sample {i+1}: R = {R_val:.4f}", end="")
                if R_conf is not None and i < len(R_conf):
                    conf_item = R_conf[i]
                    if isinstance(conf_item, (tuple, list, np.ndarray)) and len(conf_item) == 2:
                        ci_lower, ci_upper = conf_item
                        if np.isfinite(ci_lower) and np.isfinite(ci_upper):
                            print(f" (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
                        else:
                            print()
                    else:
                        print()
                else:
                    print()
            else:
                print(f"    Sample {i+1}: R = NaN (computation failed)")
        print(f"  Mean R: {np.nanmean(R_values):.4f}")
        print(f"  Std R: {np.nanstd(R_values):.4f}")
    print()


def test_compute_spectral_edge():
    """
    Test spectral edge computation from L2 norms of derivatives.
    """
    print("Testing compute_spectral_edge...")
    device = torch.device('cpu')
    
    # Create synthetic derivatives that follow ||d_n||₂ ~ Ω^n
    true_omega = 0.5
    batch_size = 5
    n_orders = 6
    dim = 20  # Dimensionality of gradients
    
    # Generate derivatives: ||d_n||₂ ~ Ω^n
    derivatives = []
    for order in range(1, n_orders + 1):
        base = torch.rand(batch_size, dim, device=device) * 0.1
        # Scale so that ||d_n||₂ ≈ Ω^n
        scale = true_omega ** order
        d_n = base * scale
        # Normalize to exact ||d_n||₂ = Ω^n (approximately)
        norms = d_n.norm(dim=1, keepdim=True)
        target_norm = true_omega ** order
        d_n = d_n / (norms + 1e-12) * target_norm
        derivatives.append(d_n)
    
    # Compute spectral edge
    omega_result = compute_spectral_edge(derivatives, normalize_by_factorial=False)
    omega_values, omega_conf = omega_result
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: L2 norms vs order
    ax1 = axes[0]
    orders = np.arange(1, n_orders + 1)
    
    for sample_idx in range(min(3, batch_size)):
        norms = []
        for d in derivatives:
            norm = d[sample_idx].norm().item()
            norms.append(norm)
        ax1.plot(orders, norms, marker='o', label=f'Sample {sample_idx+1}', alpha=0.7)
        # Show expected pattern: ||d_n||₂ ~ Ω^n
        expected_norms = norms[0] * (true_omega ** (orders - 1))
        ax1.plot(orders, expected_norms, '--', alpha=0.3)
    
    ax1.set_xlabel('Derivative Order n')
    ax1.set_ylabel('L2 Norm ||d_n||₂')
    ax1.set_title('L2 Norms of Derivatives\n(Should show exponential decay ~ Ω^n)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Estimated Ω values
    ax2 = axes[1]
    valid_mask = np.isfinite(omega_values)
    valid_omega = omega_values[valid_mask]
    sample_indices = np.arange(batch_size)[valid_mask]
    
    ax2.bar(sample_indices, valid_omega, alpha=0.7, edgecolor='black', label='Estimated Ω')
    ax2.axhline(y=true_omega, color='r', linestyle='--', linewidth=2, label=f'True Ω = {true_omega:.2f}')
    
    # Add confidence intervals
    if omega_conf is not None and len(valid_omega) > 0:
        for i, idx in enumerate(sample_indices):
            conf_item = omega_conf[idx]
            # Check if it's a tuple/array with 2 elements (confidence interval)
            if isinstance(conf_item, (tuple, list, np.ndarray)) and len(conf_item) == 2:
                ci_lower, ci_upper = conf_item
                if np.isfinite(ci_lower) and np.isfinite(ci_upper):
                    ax2.plot([idx, idx], [ci_lower, ci_upper], 'k-', linewidth=2, alpha=0.5)
    
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Spectral Edge Ω')
    ax2.set_title('Estimated Spectral Edge\n(with 95% confidence intervals)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "spectral_edge.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"\n  === Spectral Edge Results ===")
    print(f"  True Ω: {true_omega:.4f}")
    if isinstance(omega_result, tuple):
        print(f"  Estimated Ω values (batch of {batch_size}):")
        for i, omega_val in enumerate(omega_values):
            if np.isfinite(omega_val):
                print(f"    Sample {i+1}: Ω = {omega_val:.4f}", end="")
                if omega_conf is not None and i < len(omega_conf):
                    conf_item = omega_conf[i]
                    if isinstance(conf_item, (tuple, list, np.ndarray)) and len(conf_item) == 2:
                        ci_lower, ci_upper = conf_item
                        if np.isfinite(ci_lower) and np.isfinite(ci_upper):
                            print(f" (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
                        else:
                            print()
                    else:
                        print()
                else:
                    print()
            else:
                print(f"    Sample {i+1}: Ω = NaN (computation failed)")
        print(f"  Mean Ω: {np.nanmean(omega_values):.4f}")
        print(f"  Std Ω: {np.nanstd(omega_values):.4f}")
    print()


def test_detect_cyclic():
    """
    Test cyclic pattern detection in derivative sequences.
    """
    print("Testing detect_cyclic...")
    
    # Create synthetic cyclic patterns
    np.random.seed(42)
    n_orders = 8
    batch_size = 6
    
    # Pattern types:
    # 0: No cycle (random)
    # 1: 2-cycle [1, 0, 1, 0, ...]
    # 2: 4-cycle [1, 0, -1, 0, ...]
    # 3: 4-cycle shifted [0, 1, 0, -1, ...]
    # 4: 4-cycle [1, 1, -1, -1, ...]
    # 5: Another 4-cycle variant
    
    derivatives_list = []
    pattern_types = []
    
    # Sample 0: Random (no cycle)
    random_seq = np.random.randn(n_orders)
    derivatives_list.append([torch.tensor([x]) for x in random_seq])
    pattern_types.append("Random")
    
    # Sample 1: 2-cycle [1, 0, 1, 0, ...]
    cycle2 = np.array([1, 0] * (n_orders // 2 + 1))[:n_orders]
    cycle2 = cycle2 + np.random.normal(0, 0.1, n_orders)
    derivatives_list.append([torch.tensor([x]) for x in cycle2])
    pattern_types.append("2-cycle [1,0,1,0,...]")
    
    # Sample 2: 4-cycle [1, 0, -1, 0, ...]
    cycle4a = np.array([1, 0, -1, 0] * (n_orders // 4 + 1))[:n_orders]
    cycle4a = cycle4a + np.random.normal(0, 0.1, n_orders)
    derivatives_list.append([torch.tensor([x]) for x in cycle4a])
    pattern_types.append("4-cycle [1,0,-1,0,...]")
    
    # Sample 3: 4-cycle shifted [0, 1, 0, -1, ...]
    cycle4b = np.array([0, 1, 0, -1] * (n_orders // 4 + 1))[:n_orders]
    cycle4b = cycle4b + np.random.normal(0, 0.1, n_orders)
    derivatives_list.append([torch.tensor([x]) for x in cycle4b])
    pattern_types.append("4-cycle [0,1,0,-1,...]")
    
    # Sample 4: 4-cycle [1, 1, -1, -1, ...]
    cycle4c = np.array([1, 1, -1, -1] * (n_orders // 4 + 1))[:n_orders]
    cycle4c = cycle4c + np.random.normal(0, 0.1, n_orders)
    derivatives_list.append([torch.tensor([x]) for x in cycle4c])
    pattern_types.append("4-cycle [1,1,-1,-1,...]")
    
    # Sample 5: Weak cycle (borderline)
    weak_cycle = np.array([1, 0, -1, 0] * (n_orders // 4 + 1))[:n_orders]
    weak_cycle = weak_cycle + np.random.normal(0, 0.3, n_orders)  # More noise
    derivatives_list.append([torch.tensor([x]) for x in weak_cycle])
    pattern_types.append("Weak 4-cycle (noisy)")
    
    # Stack into batch format
    derivatives = []
    for order_idx in range(n_orders):
        batch_tensor = torch.stack([d[order_idx] for d in derivatives_list])
        derivatives.append(batch_tensor)
    
    # Detect cyclic patterns
    is_cyclic, cycle_period, correlation_strength = detect_cyclic(derivatives, threshold=0.7)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for sample_idx in range(batch_size):
        ax = axes[sample_idx]
        deriv_seq = np.array([d[sample_idx].item() for d in derivatives])
        orders = np.arange(1, n_orders + 1)
        
        ax.plot(orders, deriv_seq, marker='o', linewidth=2, markersize=8, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add detection info
        detected = is_cyclic[sample_idx]
        period = cycle_period[sample_idx]
        corr = correlation_strength[sample_idx]
        
        title = f"Sample {sample_idx+1}: {pattern_types[sample_idx]}"
        if detected:
            title += f"\n✓ CYCLIC (period={period:.0f}, corr={corr:.3f})"
        else:
            title += f"\n✗ Not cyclic (corr={corr:.3f})"
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Derivative Order')
        ax.set_ylabel('Derivative Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "cyclic_detection.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"\n  === Cyclic Pattern Detection Results ===")
    for i in range(batch_size):
        print(f"    Sample {i+1} ({pattern_types[i]}):")
        print(f"      Is cyclic: {is_cyclic[i]}")
        if is_cyclic[i]:
            print(f"      Period: {cycle_period[i]:.0f}")
        print(f"      Correlation strength: {correlation_strength[i]:.4f}")
    print(f"  Detected {np.sum(is_cyclic)}/{batch_size} cyclic patterns")
    print()


def test_real_model_metrics():
    """
    Test metrics on real model derivatives.
    """
    print("Testing metrics on real model...")
    device = torch.device('cpu')
    
    # Create a model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.Tanh(),
        nn.Linear(32, 10)
    ).to(device)
    
    batch_size = 4
    inputs = torch.randn(batch_size, 10, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    directions = sample_unit_directions(batch_size=batch_size, input_shape=(10,), device=device)
    
    def loss_fn(logits, labels, reduction='none'):
        return F.cross_entropy(logits, labels, reduction=reduction)
    
    max_order = 6
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
    
    # Compute all metrics
    R_result = compute_analytic_radius(derivatives)
    omega_result = compute_spectral_edge(derivatives)
    is_cyclic, period, corr = detect_cyclic(derivatives, threshold=0.6)
    
    print(f"\n  === Real Model Metrics Results ===")
    if isinstance(R_result, tuple):
        R_values, R_conf = R_result
        print(f"  Analytic Radius R: mean={np.nanmean(R_values):.4f}, std={np.nanstd(R_values):.4f}")
        for i, R_val in enumerate(R_values):
            if np.isfinite(R_val):
                print(f"    Sample {i+1}: R = {R_val:.4f}")
    
    if isinstance(omega_result, tuple):
        omega_values, omega_conf = omega_result
        print(f"  Spectral Edge Ω: mean={np.nanmean(omega_values):.4f}, std={np.nanstd(omega_values):.4f}")
        for i, omega_val in enumerate(omega_values):
            if np.isfinite(omega_val):
                print(f"    Sample {i+1}: Ω = {omega_val:.4f}")
    
    print(f"  Cyclic patterns: {np.sum(is_cyclic)}/{batch_size} detected")
    for i in range(batch_size):
        if is_cyclic[i]:
            print(f"    Sample {i+1}: period={period[i]:.0f}, correlation={corr[i]:.4f}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Metrics Functions")
    print("=" * 60)
    print()
    
    test_wls_linear()
    test_wls_linear_log()
    test_compute_analytic_radius()
    test_compute_spectral_edge()
    test_detect_cyclic()
    test_real_model_metrics()
    
    print("=" * 60)
    print("All metrics tests completed!")
    print("=" * 60)

