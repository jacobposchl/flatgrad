"""
Lambda-based regularization for neural networks.
Penalizes high-order directional derivatives to enforce flatter loss landscapes.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional
from ..derivatives import sample_unit_directions
from .lambda_estimation import compute_derivative_ratios


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


def compute_lambda_target_regularizer(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: Callable,
    target_lambda: float,
    max_order: int = 4,
    K_dirs: int = 5,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute regularization penalty that minimizes distance between current lambda and target lambda.
    
    This regularizer estimates the current lambda from derivatives and penalizes the squared
    distance from the target lambda, encouraging the model to have a specific curvature rate.
    
    Args:
        model: Neural network model
        inputs: Input batch [B, ...] - any shape
        labels: Labels [B]
        loss_fn: Loss function taking (logits, labels) with reduction='none'
        target_lambda: Target lambda value to achieve
        max_order: Maximum derivative order to use (default: 4)
        K_dirs: Number of random directions to average over (default: 5)
        scale: Scaling factor for the penalty (default: 1.0)
    
    Returns:
        Scalar penalty term to add to the loss: scale * (current_lambda - target_lambda)^2
    """
    # Save original training state and switch to eval mode
    was_training = model.training
    model.eval()
    
    batch_size = inputs.shape[0]
    input_shape = inputs.shape[1:]
    device = inputs.device
    
    # Accumulator for lambda values across directions
    lambda_values = []
    
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
        derivatives = []
        current_deriv_scalar = y_scalar
        
        for n in range(1, max_order + 1):
            # Compute gradient
            grads = torch.autograd.grad(
                current_deriv_scalar,
                inputs_with_grad,
                create_graph=True,
                retain_graph=True
            )[0]  # [B, ...]
            
            # Project onto direction: grad · direction
            grads_flat = grads.reshape(batch_size, -1)
            direction_flat = direction.reshape(batch_size, -1)
            d_n = (grads_flat * direction_flat).sum(dim=1)  # [B]
            
            derivatives.append(d_n.abs())  # Use absolute values for stability
            
            # Prepare for next iteration
            if n < max_order:
                current_deriv_scalar = d_n.sum()
        
        # Compute lambda from derivatives (differentiable version)
        if len(derivatives) >= 2:
            # Compute ratios: d_{n+1} / d_n
            ratios = []
            epsilon = 1e-12
            for i in range(len(derivatives) - 1):
                d_n = derivatives[i] + epsilon
                d_n_plus_1 = derivatives[i + 1] + epsilon
                ratio = d_n_plus_1 / d_n  # [B]
                ratios.append(ratio)
            
            # Stack ratios: [B, n_ratios]
            if len(ratios) > 0:
                ratios_stack = torch.stack(ratios, dim=1)  # [B, n_ratios]
                
                # Compute log(ratios) with numerical stability
                log_ratios = torch.log(torch.clamp(ratios_stack, min=1e-10))
                
                # Lambda is mean of log(ratios) across all ratio pairs for each sample
                lambda_per_sample = log_ratios.mean(dim=1)  # [B]
                
                # Average across batch and add to accumulator
                lambda_values.append(lambda_per_sample.mean())
    
    # Restore original training state
    if was_training:
        model.train()
    
    # If we couldn't compute any lambda values, return zero penalty
    if len(lambda_values) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Average lambda across all directions
    current_lambda = torch.stack(lambda_values).mean()
    
    # Penalize squared distance from target
    penalty = (current_lambda - target_lambda) ** 2
    
    return scale * penalty
