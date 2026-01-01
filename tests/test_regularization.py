"""
Test script to verify lambda regularization is working correctly on dummy data.

This script:
1. Creates a simple model and dummy data
2. Tests that regularization is applied per batch
3. Verifies regularization magnitude scales with reg_scale
4. Checks that regularization affects training dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flatgrad.sampling.regularizers import compute_lambda_regularizer
from flatgrad.sampling.training import train_epoch, RegTracker


def create_simple_model(input_dim=10, hidden_dim=32, output_dim=5):
    """Create a simple MLP for testing."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


def create_dummy_data(n_samples=100, input_dim=10, output_dim=5, device='cpu'):
    """Create dummy classification data."""
    inputs = torch.randn(n_samples, input_dim, device=device)
    labels = torch.randint(0, output_dim, (n_samples,), device=device)
    return inputs, labels


def test_regularization_applied():
    """Test that regularization is actually being applied."""
    print("="*60)
    print("TEST 1: Verify Regularization is Applied")
    print("="*60)
    
    device = torch.device('cpu')
    model = create_simple_model().to(device)
    
    # Create dummy data
    inputs, labels = create_dummy_data(device=device)
    
    # Test with reg_scale = 0 (no regularization)
    def regularizer_fn_zero(model, inputs, labels, loss_fn):
        return compute_lambda_regularizer(
            model, inputs, labels, loss_fn,
            start_n=1, end_n=4, K_dirs=3, scale=0.0
        )
    
    # Test with reg_scale = 1.0
    def regularizer_fn_one(model, inputs, labels, loss_fn):
        return compute_lambda_regularizer(
            model, inputs, labels, loss_fn,
            start_n=1, end_n=4, K_dirs=3, scale=1.0
        )
    
    # Compute regularization values
    reg_zero = regularizer_fn_zero(model, inputs, labels, F.cross_entropy)
    reg_one = regularizer_fn_one(model, inputs, labels, F.cross_entropy)
    
    print(f"  Regularization with scale=0.0: {reg_zero.item():.6f}")
    print(f"  Regularization with scale=1.0: {reg_one.item():.6f}")
    
    assert reg_zero.item() == 0.0, "Regularization with scale=0 should be 0"
    assert reg_one.item() > 0.0, "Regularization with scale=1.0 should be positive"
    
    print("  ✓ PASS: Regularization is being applied correctly\n")


def test_regularization_scaling():
    """Test that regularization scales linearly with reg_scale."""
    print("="*60)
    print("TEST 2: Verify Regularization Scales with reg_scale")
    print("="*60)
    
    device = torch.device('cpu')
    model = create_simple_model().to(device)
    
    # Create dummy data
    inputs, labels = create_dummy_data(device=device)
    
    reg_scales = [0.1, 1.0, 10.0, 100.0]
    reg_values = []
    
    for scale in reg_scales:
        def regularizer_fn(model, inputs, labels, loss_fn):
            return compute_lambda_regularizer(
                model, inputs, labels, loss_fn,
                start_n=1, end_n=4, K_dirs=3, scale=scale
            )
        
        reg_value = regularizer_fn(model, inputs, labels, F.cross_entropy)
        reg_values.append(reg_value.item())
        print(f"  Scale={scale:6.1f}: Reg Loss={reg_value.item():.6f}")
    
    # Check that values scale approximately linearly
    ratios = [reg_values[i] / reg_values[0] for i in range(1, len(reg_values))]
    expected_ratios = [reg_scales[i] / reg_scales[0] for i in range(1, len(reg_scales))]
    
    print(f"\n  Ratios (actual): {[f'{r:.2f}' for r in ratios]}")
    print(f"  Ratios (expected): {[f'{r:.2f}' for r in expected_ratios]}")
    
    # Allow some tolerance for numerical differences
    for actual, expected in zip(ratios, expected_ratios):
        assert abs(actual - expected) / expected < 0.2, \
            f"Regularization should scale linearly (got ratio {actual:.2f}, expected {expected:.2f})"
    
    print("  ✓ PASS: Regularization scales correctly with reg_scale\n")


def test_regularization_per_batch():
    """Test that regularization is applied per batch during training."""
    print("="*60)
    print("TEST 3: Verify Regularization Applied Per Batch")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Create model and data
    model = create_simple_model().to(device)
    inputs1, labels1 = create_dummy_data(n_samples=32, device=device)
    inputs2, labels2 = create_dummy_data(n_samples=32, device=device)
    
    # Create data loader with 2 batches
    dataset = TensorDataset(torch.cat([inputs1, inputs2]), torch.cat([labels1, labels2]))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create regularizer
    reg_scale = 1.0
    def regularizer_fn(model, inputs, labels, loss_fn):
        return compute_lambda_regularizer(
            model, inputs, labels, loss_fn,
            start_n=1, end_n=4, K_dirs=3, scale=reg_scale
        )
    
    # Track regularization
    reg_tracker = RegTracker()
    
    # Train for one epoch
    train_metrics = train_epoch(
        model, train_loader, optimizer,
        F.cross_entropy, device, regularizer_fn
    )
    
    # Check that regularization was tracked
    assert 'reg_loss' in train_metrics, "train_epoch should return reg_loss"
    assert 'main_loss' in train_metrics, "train_epoch should return main_loss"
    assert train_metrics['reg_loss'] > 0, "Regularization loss should be positive"
    
    print(f"  Main Loss: {train_metrics['main_loss']:.6f}")
    print(f"  Reg Loss: {train_metrics['reg_loss']:.6f}")
    print(f"  Total Loss: {train_metrics['loss']:.6f}")
    print(f"  Reg Ratio: {train_metrics['reg_loss'] / train_metrics['main_loss']:.6f}")
    
    # Verify total loss = main_loss + reg_loss
    assert abs(train_metrics['loss'] - (train_metrics['main_loss'] + train_metrics['reg_loss'])) < 1e-6, \
        "Total loss should equal main_loss + reg_loss"
    
    print("  ✓ PASS: Regularization is applied per batch correctly\n")


def test_regularization_affects_training():
    """Test that regularization actually affects training dynamics."""
    print("="*60)
    print("TEST 4: Verify Regularization Affects Training")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Create two identical models
    model_no_reg = create_simple_model().to(device)
    model_with_reg = create_simple_model().to(device)
    
    # Copy weights to make them identical
    model_with_reg.load_state_dict(model_no_reg.state_dict())
    
    # Create data
    inputs, labels = create_dummy_data(n_samples=100, device=device)
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Train model without regularization
    optimizer_no_reg = optim.Adam(model_no_reg.parameters(), lr=0.001)
    for _ in range(5):
        train_epoch(model_no_reg, train_loader, optimizer_no_reg, 
                   F.cross_entropy, device, None)
    
    # Train model with regularization
    reg_scale = 10.0
    def regularizer_fn(model, inputs, labels, loss_fn):
        return compute_lambda_regularizer(
            model, inputs, labels, loss_fn,
            start_n=1, end_n=4, K_dirs=3, scale=reg_scale
        )
    
    optimizer_with_reg = optim.Adam(model_with_reg.parameters(), lr=0.001)
    reg_tracker = RegTracker()
    
    for epoch in range(5):
        train_metrics = train_epoch(model_with_reg, train_loader, optimizer_with_reg,
                                    F.cross_entropy, device, regularizer_fn)
        reg_tracker.record(epoch + 1, train_metrics['reg_loss'], train_metrics['main_loss'])
    
    # Check that models have diverged (regularization should affect training)
    params_no_reg = [p.data.clone() for p in model_no_reg.parameters()]
    params_with_reg = [p.data.clone() for p in model_with_reg.parameters()]
    
    max_diff = max([(p1 - p2).abs().max().item() 
                   for p1, p2 in zip(params_no_reg, params_with_reg)])
    
    print(f"  Max parameter difference: {max_diff:.6f}")
    print(f"  Mean reg ratio: {np.mean(reg_tracker.get_history()['reg_ratios']):.6f}")
    
    assert max_diff > 1e-6, "Models should diverge when regularization is applied"
    
    print("  ✓ PASS: Regularization affects training dynamics\n")


def test_reg_tracker():
    """Test that RegTracker works correctly."""
    print("="*60)
    print("TEST 5: Verify RegTracker Functionality")
    print("="*60)
    
    tracker = RegTracker()
    
    # Record some values
    tracker.record(1, 0.01, 0.5)
    tracker.record(2, 0.02, 0.4)
    tracker.record(3, 0.015, 0.35)
    
    history = tracker.get_history()
    
    assert len(history['epochs']) == 3
    assert len(history['reg_losses']) == 3
    assert len(history['main_losses']) == 3
    assert len(history['reg_ratios']) == 3
    
    # Check ratios
    expected_ratios = [0.01/0.5, 0.02/0.4, 0.015/0.35]
    for actual, expected in zip(history['reg_ratios'], expected_ratios):
        assert abs(actual - expected) < 1e-6, f"Ratio mismatch: {actual} vs {expected}"
    
    print(f"  Epochs: {history['epochs']}")
    print(f"  Reg Losses: {[f'{r:.4f}' for r in history['reg_losses']]}")
    print(f"  Main Losses: {[f'{m:.4f}' for m in history['main_losses']]}")
    print(f"  Reg Ratios: {[f'{r:.4f}' for r in history['reg_ratios']]}")
    
    tracker.print_summary()
    
    print("  ✓ PASS: RegTracker works correctly\n")


def main():
    """Run all regularization tests."""
    print("\n" + "="*60)
    print("REGULARIZATION TEST SUITE")
    print("="*60 + "\n")
    
    try:
        test_regularization_applied()
        test_regularization_scaling()
        test_regularization_per_batch()
        test_regularization_affects_training()
        test_reg_tracker()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    main()

