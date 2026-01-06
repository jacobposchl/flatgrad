'''
modules for n-th order derivatives
'''

import torch
from typing import List, Tuple, Callable

def sample_unit_directions(
    batch_size: int, 
    input_shape: Tuple[int, ...], 
    device: torch.device
) -> torch.Tensor:
    """
    Sample random unit-norm directions in input space.
        
    Args:
        batch_size: Number of directions to sample
        input_shape: Shape of a single input (excluding batch dimension)
                    e.g., (1, 28, 28) for MNIST, (784,) for flattened vectors
        device: Device to create tensors on
    
    Returns:
        Tensor of shape [batch_size, *input_shape] containing unit-norm directions
    """

    # Sample Gaussian random directions
    U = torch.randn(batch_size, *input_shape, device=device)
    
    # Flatten to compute norms, then reshape back
    U_flat = U.view(batch_size, -1)
    norms = U_flat.norm(dim=1, keepdim=True)
    U_flat = U_flat / (norms + 1e-12)  # Normalize to unit length (add epsilon for numerical stability)
    
    return U_flat.view(batch_size, *input_shape)


def compute_directional_derivatives(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    directions: torch.Tensor,
    loss_fn: Callable,
    min_order: int,
    max_order: int,
    create_graph: bool = True,
) -> List[torch.Tensor]:
    """
    Compute directional derivatives of the loss function from min_order to max_order.
    
    Note: Derivatives are computed sequentially from 1 to max_order, but only
    those from min_order to max_order are returned.
    
    Args:
        model: Neural network model
        inputs: Input batch [B, ...] - any shape
        labels: Labels for the batch [B]
        directions: Unit directions [B, ...] - must match input shape
        loss_fn: Loss function that takes (logits, labels) and returns loss
                 Should support reduction='none' to return per-sample losses
        min_order: Minimum derivative order to return (must be >= 1)
        max_order: Maximum derivative order to compute
        create_graph: Whether to create computation graph (needed for higher-order gradients)
    
    Returns:
        List of tensors, where derivatives[i] is the (min_order+i)-th order derivative [B]
        Length of list is (max_order - min_order + 1)
    """

    # Input validation
    if min_order < 1:
        raise ValueError(f"min_order must be >= 1, got {min_order}")
    if max_order < min_order:
        raise ValueError(f"max_order ({max_order}) must be >= min_order ({min_order})")
    if inputs.shape[0] != directions.shape[0]:
        raise ValueError(f"Batch size mismatch: inputs has {inputs.shape[0]}, directions has {directions.shape[0]}")
    if inputs.shape[1:] != directions.shape[1:]:
        raise ValueError(f"Shape mismatch: inputs shape {inputs.shape[1:]}, directions shape {directions.shape[1:]}")
    
    # Ensure inputs require gradients and are on the correct device
    inputs = inputs.to(directions.device).requires_grad_(True)
    labels = labels.to(directions.device)
    
    # Forward pass through model
    logits = model(inputs)
    loss_per_sample = loss_fn(logits, labels)  # [B]
    loss_scalar = loss_per_sample.sum()  # Scalar for autograd
    
    all_derivatives = []
    
    # Compute first-order derivative
    # Compute gradient w.r.t. inputs
    grads = torch.autograd.grad(
        loss_scalar,
        inputs,
        create_graph=(max_order > 1 and create_graph),  # Only create graph if we need higher orders
        retain_graph=True
    )[0]  # [B, ...]
    
    # Compute directional derivative: grad * direction
    # Flatten both to compute dot product, then sum over spatial dimensions
    grads_flat = grads.view(grads.size(0), -1)  # [B, D]
    dirs_flat = directions.view(directions.size(0), -1)  # [B, D]
    d_1 = (grads_flat * dirs_flat).sum(dim=1)  # [B]
    
    all_derivatives.append(d_1)
    
    # Compute higher-order derivatives sequentially from 2 to max_order
    if max_order > 1:
        if not create_graph:
            raise ValueError("create_graph must be True to compute derivatives of order > 1")
        
        current_derivative = d_1
        for n in range(2, max_order + 1):
            # Check if current_derivative has grad_fn (it might be zero/constant for high orders)
            if not hasattr(current_derivative, 'grad_fn') or current_derivative.grad_fn is None:
                # Derivative has become constant (likely zero for polynomial beyond its degree)
                # All higher derivatives will also be zero
                zero_deriv = torch.zeros_like(current_derivative)
                all_derivatives.append(zero_deriv)
                current_derivative = zero_deriv
                continue
            
            # Compute gradient of the previous directional derivative w.r.t. inputs
            # We need to sum the current derivative to get a scalar for backprop
            grads = torch.autograd.grad(
                current_derivative.sum(),
                inputs,
                create_graph=True,  # Always True for higher order derivatives
                retain_graph=True
            )[0]  # [B, ...]
            
            # Compute directional derivative of this gradient
            grads_flat = grads.view(grads.size(0), -1)  # [B, D]
            d_n = (grads_flat * dirs_flat).sum(dim=1)  # [B]
            
            all_derivatives.append(d_n)
            current_derivative = d_n
    
    # Return only the derivatives in the requested range
    return all_derivatives[min_order - 1:max_order]

