import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flatgrad.derivatives import sample_unit_directions, compute_directional_derivatives
from configs.test_config import TEST_DERIVATIVES_OUTPUT_DIR as OUTPUT_DIR

def test_and_plot_2d_unit_vectors():
    """
    Visualize 2D unit vectors to demonstrate the unit norm concept.
    Shows vectors on the unit circle (all with length = 1).
    """
    print("Visualizing 2D unit vectors...")
    device = torch.device('cpu')
    
    # Sample 2D unit vectors
    batch_size = 30
    directions = sample_unit_directions(batch_size=batch_size, input_shape=(2,), device=device)
    
    # Convert to numpy for plotting
    vectors = directions.numpy()  # Shape: (30, 2)
    x_coords = vectors[:, 0]
    y_coords = vectors[:, 1]
    
    # Compute norms to verify
    norms = np.sqrt(x_coords**2 + y_coords**2)
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Unit circle with vectors
    ax1.set_aspect('equal')
    
    # Draw the unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--', linewidth=2, label='Unit circle (radius=1)')
    ax1.add_patch(circle)
    
    # Plot vectors as arrows from origin
    for i in range(batch_size):
        ax1.arrow(0, 0, x_coords[i], y_coords[i], 
                 head_width=0.05, head_length=0.05, 
                 fc='blue', ec='blue', alpha=0.6, length_includes_head=True)
    
    # Mark the origin
    ax1.plot(0, 0, 'ko', markersize=8, label='Origin')
    
    # Add some example unit vectors as reference
    ref_vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for vx, vy in ref_vectors:
        ax1.arrow(0, 0, vx, vy, head_width=0.08, head_length=0.08,
                 fc='green', ec='green', alpha=0.8, linewidth=2, length_includes_head=True)
    
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_title(f'2D Unit Vectors on Unit Circle\n({batch_size} random vectors, all ||v|| ≈ 1.0)')
    ax1.legend(loc='upper right')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Plot 2: Distribution of angles
    angles = np.arctan2(y_coords, x_coords)  # Angle in radians
    angles_degrees = np.degrees(angles)
    
    ax2.hist(angles_degrees, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Vector Angles')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-180, 180)
    
    # Add reference lines for cardinal directions
    for angle_deg in [0, 90, -90, 180]:
        ax2.axvline(x=angle_deg, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # Save plot
    output_path = OUTPUT_DIR / "2d_unit_vectors.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"  Mean norm: {norms.mean():.10f} (should be ~1.0)")
    print(f"  Std norm: {norms.std():.2e} (should be ~0.0)")
    print(f"  Min norm: {norms.min():.10f}")
    print(f"  Max norm: {norms.max():.10f}")
    print(f"  All on unit circle: {np.allclose(norms, 1.0, atol=1e-5)}\n")

def test_and_plot_MNIST_directions():
    """
    Test and visualize that sampled directions have unit norm.
    """
    print("Testing sample_unit_directions")
    device = torch.device('cpu')
    
    # Sample directions
    batch_size = 20
    directions = sample_unit_directions(batch_size=batch_size, input_shape=(784,), device=device)
    
    # Check norms
    norms = directions.view(batch_size, -1).norm(dim=1).numpy()
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(batch_size), norms, alpha=0.7, edgecolor='black')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Target: Unit norm (1.0)')
    plt.xlabel('Direction index')
    plt.ylabel('Norm')
    plt.title(f'Direction Norms (Mean: {norms.mean():.10f}, Std: {norms.std():.2e})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.999999, 1.000001)  # Zoom in to see small variations
    
    plt.tight_layout()
    
    # Save plot
    output_path = OUTPUT_DIR / "directions_norms.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"  Mean norm: {norms.mean():.10f} (should be ~1.0)")
    print(f"  Std norm: {norms.std():.2e} (should be ~0.0)")
    print(f"  Min norm: {norms.min():.10f}")
    print(f"  Max norm: {norms.max():.10f}")
    print(f"  All close to 1.0: {np.allclose(norms, 1.0, atol=1e-5)}\n")


def test_and_plot_derivatives():
    """
    Test and visualize directional derivative computation.
    """
    print("Testing compute_directional_derivatives")

    device = torch.device('cpu')
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    ).to(device)
    
    # Create dummy data
    batch_size = 5
    inputs = torch.randn(batch_size, 10, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    directions = sample_unit_directions(batch_size=batch_size, input_shape=(10,), device=device)
    
    def loss_fn(logits, labels, reduction='none'):
        return F.cross_entropy(logits, labels, reduction=reduction)
    
    # Compute derivatives (1st through 5th order)
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
    
    # Convert to numpy for plotting
    orders = list(range(1, max_order + 1))
    deriv_values = [d.detach().numpy() for d in derivatives]
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i in range(batch_size):
        values = [d[i] for d in deriv_values]
        plt.plot(orders, values, marker='o', label=f'Sample {i+1}')
    plt.xlabel('Derivative Order')
    plt.ylabel('Derivative Value')
    plt.title(f'Directional Derivatives (n={max_order} orders)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('symlog')  # Use symlog to handle potentially large values
    
    plt.subplot(1, 2, 2)
    mean_derivs = [d.mean().item() for d in derivatives]
    std_derivs = [d.std().item() for d in derivatives]
    plt.errorbar(orders, mean_derivs, yerr=std_derivs, marker='o', capsize=5, capthick=2)
    plt.xlabel('Derivative Order')
    plt.ylabel('Mean Derivative Value')
    plt.title('Mean ± Std Across Batch')
    plt.grid(True, alpha=0.3)
    plt.yscale('symlog')
    
    plt.tight_layout()
    
    # Save plot
    output_path = OUTPUT_DIR / "derivatives.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    print(f"  Computed {len(derivatives)} derivative orders (1-{max_order})")
    print(f"  Each derivative shape: {derivatives[0].shape}")
    print(f"  Mean absolute values by order:")
    for i, deriv in enumerate(derivatives):
        print(f"    Order {i+1}: {deriv.abs().mean().item():.6f}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Derivative Functions")
    print("=" * 60 + "\n")
    test_and_plot_2d_unit_vectors()
    test_and_plot_MNIST_directions()
    test_and_plot_derivatives()