import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flatgrad.derivatives import sample_unit_directions, compute_directional_derivatives
from flatgrad.sampling.lambda_estimation import estimate_lambda_from_derivatives
from flatgrad.sampling.metrics import (
    compute_analytic_radius,
    compute_spectral_edge,
    detect_cyclic
)
from flatgrad.sampling.models import (
    PolynomialModel,
    QuadraticModel,
    SimpleMLP,
    ExponentialDecayModel,
    SinusoidalModel,
    LinearCombinationModel,
    create_test_model,
    compute_analytical_lambda,
    compute_analytical_radius as compute_analytical_radius_from_model
)
from configs.test_config import TEST_MODELS_OUTPUT_DIR as OUTPUT_DIR


def test_polynomial_model():
    """
    Test polynomial model with known analytical properties.
    
    Note: For a degree-2 polynomial model f(x) with MSE loss L = (f(x) - y)^2,
    the loss itself becomes a degree-4 polynomial, so only derivatives of order > 4
    should be zero.
    """
    print("Testing PolynomialModel...")
    device = torch.device('cpu')
    
    # Create a quadratic polynomial (degree 2)
    input_dim = 10
    output_dim = 1
    degree = 2
    
    model = PolynomialModel(degree=degree, input_dim=input_dim, output_dim=output_dim).to(device)
    model.eval()
    
    batch_size = 4
    inputs = torch.randn(batch_size, input_dim, device=device)
    labels = torch.randn(batch_size, device=device)  # For MSE loss
    directions = sample_unit_directions(batch_size=batch_size, input_shape=(input_dim,), device=device)
    
    # Forward pass test
    outputs = model(inputs)
    print(f"  Model output shape: {outputs.shape} (expected: [{batch_size}, {output_dim}])")
    
    # Test derivatives - MSE loss squares the polynomial, making it degree 2*N
    def loss_fn(logits, labels, reduction='none'):
        # For regression, use MSE
        if logits.shape[1] == 1:
            return F.mse_loss(logits.squeeze(1), labels, reduction=reduction)
        else:
            return F.mse_loss(logits, labels.unsqueeze(1).expand(-1, logits.shape[1]), reduction=reduction)
    
    max_order = 6
    loss_degree = 2 * degree  # MSE makes loss degree = 2 * model_degree
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
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Derivatives by order
    ax1 = axes[0]
    orders = list(range(1, max_order + 1))
    derivatives_np = [d.detach().cpu().numpy() for d in derivatives]
    
    for i in range(batch_size):
        deriv_values = [d[i] for d in derivatives_np]
        ax1.plot(orders, deriv_values, marker='o', label=f'Sample {i+1}', alpha=0.7)
    
    # Mark where derivatives should become zero (MSE makes loss degree 2*N)
    ax1.axvline(x=loss_degree + 1, color='r', linestyle='--', linewidth=2, 
                label=f'Order {loss_degree + 1} (should be ~0 for degree {loss_degree} loss)')
    
    ax1.set_xlabel('Derivative Order')
    ax1.set_ylabel('Derivative Value')
    ax1.set_title(f'Polynomial Model (degree={degree}) + MSE Loss (degree={loss_degree})\n(Order > {loss_degree} should be ~0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('symlog')
    
    # Plot 2: Check if higher-order derivatives are near zero
    ax2 = axes[1]
    order_mean_abs = [d.abs().mean().item() for d in derivatives]
    ax2.bar(orders, order_mean_abs, alpha=0.7, edgecolor='black')
    ax2.axvline(x=loss_degree + 1, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Derivative Order')
    ax2.set_ylabel('Mean Absolute Value')
    ax2.set_title(f'Mean |Derivative| by Order\n(Should drop to ~0 after order {loss_degree})')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "polynomial_model.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"  Derivative magnitudes by order:")
    for order, deriv in enumerate(derivatives, 1):
        mean_abs = deriv.abs().mean().item()
        print(f"    Order {order}: {mean_abs:.2e}")
        if order > loss_degree:
            print(f"      (Expected ~0 for order > {loss_degree})")
    print()


def test_quadratic_model():
    """
    Test quadratic model (specialized polynomial).
    """
    print("Testing QuadraticModel...")
    device = torch.device('cpu')
    
    input_dim = 10
    model = QuadraticModel(input_dim=input_dim, output_dim=1).to(device)
    model.eval()
    
    batch_size = 3
    inputs = torch.randn(batch_size, input_dim, device=device)
    labels = torch.randn(batch_size, device=device)
    directions = sample_unit_directions(batch_size=batch_size, input_shape=(input_dim,), device=device)
    
    def loss_fn(logits, labels, reduction='none'):
        return F.mse_loss(logits.squeeze(1), labels, reduction=reduction)
    
    max_order = 4
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
    
    print(f"  Quadratic model (degree=2):")
    print(f"    Order 3 derivative mean: {derivatives[2].abs().mean().item():.2e} (should be ~0)")
    if max_order >= 4:
        print(f"    Order 4 derivative mean: {derivatives[3].abs().mean().item():.2e} (should be ~0)")
    print()


def test_simple_mlp():
    """
    Test SimpleMLP model.
    """
    print("Testing SimpleMLP...")
    device = torch.device('cpu')
    
    # Create MLP with different activations
    input_dim = 10
    hidden_dims = [32, 16]
    output_dim = 10
    
    model_tanh = SimpleMLP(input_dim, hidden_dims, output_dim, activation='tanh').to(device)
    model_relu = SimpleMLP(input_dim, hidden_dims, output_dim, activation='relu').to(device)
    
    batch_size = 4
    inputs = torch.randn(batch_size, input_dim, device=device)
    labels = torch.randint(0, output_dim, (batch_size,), device=device)
    directions = sample_unit_directions(batch_size=batch_size, input_shape=(input_dim,), device=device)
    
    def loss_fn(logits, labels, reduction='none'):
        return F.cross_entropy(logits, labels, reduction=reduction)
    
    # Test both activations
    for activation, model in [('tanh', model_tanh), ('relu', model_relu)]:
        model.eval()
        max_order = 4
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
        
        lambda_val = estimate_lambda_from_derivatives(derivatives, abs_derivatives=True)
        print(f"  MLP ({activation}): lambda = {lambda_val:.4f}" if lambda_val else f"  MLP ({activation}): lambda = None")
    
    print()


def test_exponential_decay_model():
    """
    Test ExponentialDecayModel with known analytical lambda.
    """
    print("Testing ExponentialDecayModel...")
    device = torch.device('cpu')
    
    input_dim = 10
    output_dim = 1
    decay_factor = 0.5
    true_lambda = np.log(decay_factor)
    true_R = 1.0 / decay_factor
    
    model = ExponentialDecayModel(
        input_dim=input_dim,
        output_dim=output_dim,
        decay_factor=decay_factor
    ).to(device)
    model.eval()
    
    batch_size = 5
    inputs = torch.randn(batch_size, input_dim, device=device)
    labels = torch.randn(batch_size, device=device)
    directions = sample_unit_directions(batch_size=batch_size, input_shape=(input_dim,), device=device)
    
    def loss_fn(logits, labels, reduction='none'):
        return F.mse_loss(logits.squeeze(1), labels, reduction=reduction)
    
    max_order = 5
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
    
    # Get analytical values
    analytical_lambda = compute_analytical_lambda(model, inputs, directions)
    analytical_R = compute_analytical_radius_from_model(model, inputs, directions)
    
    # Estimate from derivatives
    estimated_lambda = estimate_lambda_from_derivatives(derivatives, abs_derivatives=True)
    R_result = compute_analytic_radius(derivatives)
    if isinstance(R_result, tuple):
        R_values, _ = R_result
        estimated_R = np.nanmean(R_values)
    else:
        estimated_R = R_result
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Derivatives showing exponential decay
    ax1 = axes[0]
    orders = list(range(1, max_order + 1))
    derivatives_np = [d.detach().cpu().numpy() for d in derivatives]
    
    for i in range(min(3, batch_size)):
        deriv_values = [np.abs(d[i]) for d in derivatives_np]
        ax1.plot(orders, deriv_values, marker='o', label=f'Sample {i+1}', alpha=0.7)
    
    # Show expected exponential decay
    if len(derivatives_np) > 0:
        first_deriv = np.abs(derivatives_np[0][0])
        expected_decay = [first_deriv * (decay_factor ** (n - 1)) for n in orders]
        ax1.plot(orders, expected_decay, 'r--', linewidth=2, label=f'Expected (λ={true_lambda:.3f})')
    
    ax1.set_xlabel('Derivative Order')
    ax1.set_ylabel('|Derivative Value|')
    ax1.set_title(f'ExponentialDecayModel Derivatives\n(Should show exponential decay ~{decay_factor}^n)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Lambda comparison
    ax2 = axes[1]
    if analytical_lambda is not None and estimated_lambda is not None:
        x_pos = [0, 1]
        values = [analytical_lambda, estimated_lambda]
        labels = ['Analytical', 'Estimated']
        colors = ['green', 'blue']
        bars = ax2.bar(x_pos, values, alpha=0.7, edgecolor='black', color=colors)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('λ Value')
        ax2.set_title('Lambda Comparison\n(Green=Analytical, Blue=Estimated)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "exponential_decay_model.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"\n  === ExponentialDecayModel Results ===")
    print(f"  True lambda (analytical): {analytical_lambda:.4f}")
    print(f"  Estimated lambda: {estimated_lambda:.4f}" if estimated_lambda else "  Estimated lambda: None")
    if analytical_lambda is not None and estimated_lambda is not None:
        error = abs(analytical_lambda - estimated_lambda)
        print(f"  Error: {error:.4f}")
    
    print(f"  True R (analytical): {analytical_R:.4f}")
    if estimated_R is not None and np.isfinite(estimated_R):
        print(f"  Estimated R: {estimated_R:.4f}")
        error_R = abs(analytical_R - estimated_R)
        print(f"  Error: {error_R:.4f}")
    print()


def test_sinusoidal_model():
    """
    Test SinusoidalModel with cyclic pattern detection.
    """
    print("Testing SinusoidalModel...")
    device = torch.device('cpu')
    
    input_dim = 10
    output_dim = 1
    frequency = 2.0  # Higher frequency for more pronounced cyclic patterns
    
    model = SinusoidalModel(input_dim=input_dim, output_dim=output_dim, frequency=frequency).to(device)
    model.eval()
    
    batch_size = 10  # More samples for better detection
    inputs = torch.randn(batch_size, input_dim, device=device)
    labels = torch.randn(batch_size, device=device)
    directions = sample_unit_directions(batch_size=batch_size, input_shape=(input_dim,), device=device)
    
    def loss_fn(logits, labels, reduction='none'):
        return F.mse_loss(logits.squeeze(1), labels, reduction=reduction)
    
    max_order = 8  # Need more orders to see cycle
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
    
    # Detect cyclic patterns - don't use abs for sign-based cycles!
    is_cyclic, cycle_period, correlation = detect_cyclic(derivatives, threshold=0.5, abs_derivatives=False)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Derivatives showing cyclic pattern
    ax1 = axes[0]
    orders = list(range(1, max_order + 1))
    derivatives_np = [d.detach().cpu().numpy() for d in derivatives]
    
    for i in range(min(3, batch_size)):
        deriv_values = [d[i] for d in derivatives_np]
        detected = is_cyclic[i] if i < len(is_cyclic) else False
        period = cycle_period[i] if i < len(cycle_period) else np.nan
        label = f'Sample {i+1}'
        if detected:
            label += f' (cyclic, period={period:.0f})'
        ax1.plot(orders, deriv_values, marker='o', label=label, alpha=0.7)
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Derivative Order')
    ax1.set_ylabel('Derivative Value')
    ax1.set_title('SinusoidalModel Derivatives\n(Should show oscillatory/cyclic pattern)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cyclic detection results
    ax2 = axes[1]
    if len(is_cyclic) > 0:
        detected_count = np.sum(is_cyclic)
        ax2.bar(['Cyclic', 'Not Cyclic'], 
                [detected_count, len(is_cyclic) - detected_count],
                alpha=0.7, edgecolor='black', color=['green', 'red'])
        ax2.set_ylabel('Count')
        ax2.set_title(f'Cyclic Pattern Detection\n({detected_count}/{len(is_cyclic)} detected)')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "sinusoidal_model.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"\n  === SinusoidalModel Results ===")
    print(f"  Cyclic patterns detected: {np.sum(is_cyclic)}/{len(is_cyclic)}")
    for i in range(len(is_cyclic)):
        if is_cyclic[i]:
            print(f"    Sample {i+1}: period={cycle_period[i]:.0f}, correlation={correlation[i]:.4f}")
    print()


def test_linear_combination_model():
    """
    Test LinearCombinationModel (composite model).
    """
    print("Testing LinearCombinationModel...")
    device = torch.device('cpu')
    
    input_dim = 10
    output_dim = 1
    
    model = LinearCombinationModel(input_dim=input_dim, output_dim=output_dim).to(device)
    model.eval()
    
    batch_size = 4
    inputs = torch.randn(batch_size, input_dim, device=device)
    labels = torch.randn(batch_size, device=device)
    directions = sample_unit_directions(batch_size=batch_size, input_shape=(input_dim,), device=device)
    
    def loss_fn(logits, labels, reduction='none'):
        return F.mse_loss(logits.squeeze(1), labels, reduction=reduction)
    
    max_order = 5
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
    
    # Compute metrics
    lambda_val = estimate_lambda_from_derivatives(derivatives, abs_derivatives=True)
    R_result = compute_analytic_radius(derivatives)
    omega_result = compute_spectral_edge(derivatives)
    is_cyclic, period, corr = detect_cyclic(derivatives, threshold=0.6, abs_derivatives=False)
    
    print(f"\n  === LinearCombinationModel Results ===")
    print(f"  Estimated lambda: {lambda_val:.4f}" if lambda_val else "  Estimated lambda: None")
    
    if isinstance(R_result, tuple):
        R_values, _ = R_result
        print(f"  Estimated R: {np.nanmean(R_values):.4f}")
    
    if isinstance(omega_result, tuple):
        omega_values, _ = omega_result
        print(f"  Estimated Ω: {np.nanmean(omega_values):.4f}")
    
    print(f"  Cyclic patterns: {np.sum(is_cyclic)}/{len(is_cyclic)} detected")
    print()


def test_create_test_model_factory():
    """
    Test the factory function for creating models.
    """
    print("Testing create_test_model factory...")
    
    input_dim = 10
    output_dim = 5
    
    # Test each model type
    model_types = ['polynomial', 'quadratic', 'mlp', 'exponential', 'sinusoidal']
    
    for model_type in model_types:
        try:
            if model_type == 'polynomial':
                model = create_test_model(model_type, input_dim, output_dim, degree=3)
            elif model_type == 'mlp':
                model = create_test_model(model_type, input_dim, output_dim, hidden_dims=[32, 16], activation='tanh')
            elif model_type == 'exponential':
                model = create_test_model(model_type, input_dim, output_dim, decay_factor=0.6)
            else:
                model = create_test_model(model_type, input_dim, output_dim)
            
            # Test forward pass
            test_input = torch.randn(2, input_dim)
            output = model(test_input)
            assert output.shape == (2, output_dim), f"Wrong output shape for {model_type}"
            
            print(f"  ✓ {model_type}: created and forward pass works")
        except Exception as e:
            print(f"  ✗ {model_type}: failed with {e}")
    
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Test Models")
    print("=" * 60)
    print()
    
    test_polynomial_model()
    test_quadratic_model()
    test_simple_mlp()
    test_exponential_decay_model()
    test_sinusoidal_model()
    test_linear_combination_model()
    test_create_test_model_factory()
    
    print("=" * 60)
    print("All model tests completed!")
    print("=" * 60)

