"""
Lambda-based regularization for neural networks.
Penalizes high-order directional derivatives to enforce flatter loss landscapes.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional
from ..derivatives import sample_unit_directions


def compute_lambda_regularizer(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: Callable,
    start_n: int = 1,
    end_n: int = 4,
    K_dirs: int = 5,
    scale: float = 1e-3,
) -> torch.Tensor:
    """
    Compute lambda regularization penalty by penalizing nth-order directional derivatives.
    
    This regularizer encourages the model to have lower curvature rates (smaller λ)
    by directly penalizing high-order derivatives of the loss function.
    
    Args:
        model: Neural network model
        inputs: Input batch [B, ...] - any shape
        labels: Labels [B]
        loss_fn: Loss function taking (logits, labels) with reduction='none'
        start_n: Starting derivative order to penalize (default: 1)
        end_n: Ending derivative order to penalize (default: 4)
        K_dirs: Number of random directions to average over (default: 5)
        scale: Scaling factor for the penalty (default: 1e-3)
    
    Returns:
        Scalar penalty term to add to the loss
    """
    # Save original training state and switch to eval mode
    # This ensures deterministic forward passes (no dropout randomness)
    was_training = model.training
    model.eval()
    
    batch_size = inputs.shape[0]
    input_shape = inputs.shape[1:]
    device = inputs.device
    
    # Accumulator for penalty across directions
    total_penalty = 0.0
    
    # Average over multiple random directions
    for _ in range(K_dirs):
        # Sample random unit direction
        direction = sample_unit_directions(batch_size, input_shape, device)  # [B, ...]
        
        # Enable gradients on inputs
        inputs_with_grad = inputs.detach().requires_grad_(True)
        
        # Forward pass
        logits = model(inputs_with_grad)
        loss_per_sample = loss_fn(logits, labels, reduction='none')  # [B]
        y_scalar = loss_per_sample.sum()
        
        # Compute derivatives sequentially
        current_deriv_scalar = y_scalar
        
        for n in range(1, end_n + 1):
            # Compute gradient
            grads = torch.autograd.grad(
                current_deriv_scalar,
                inputs_with_grad,
                create_graph=True,
                retain_graph=True
            )[0]  # [B, ...]
            
            # Project onto direction: grad · direction
            # Flatten both and compute dot product
            grads_flat = grads.reshape(batch_size, -1)
            direction_flat = direction.reshape(batch_size, -1)
            d_n = (grads_flat * direction_flat).sum(dim=1)  # [B]
            
            # Add penalty if within specified range
            if n >= start_n:
                # Penalize absolute value of derivative
                total_penalty += d_n.abs().mean()
            
            # Prepare for next iteration
            if n < end_n:
                current_deriv_scalar = d_n.sum()
    
    # Average across directions
    avg_penalty = total_penalty / K_dirs
    
    # Restore original training state
    if was_training:
        model.train()
    
    return scale * avg_penalty
