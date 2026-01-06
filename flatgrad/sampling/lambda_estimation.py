'''
Estimate Lambda:
Calculated as \\lambda = log(\\frac{d_{n+1}}{d_n}) where d_n is the n-th order derivative
'''

import torch
import numpy as np
from typing import List, Optional, Callable

from ..derivatives import compute_directional_derivatives, sample_unit_directions


def compute_derivative_ratios(derivatives: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute ratios between consecutive derivative orders.
    
    Args:
        derivatives: List of [B] tensors, one per derivative order
                    derivatives[i] is the (i+1)-th order derivative
        
    Returns:
        ratios: Tensor of shape [B, n_ratios] where n_ratios = len(derivatives) - 1
                ratios[b, i] = derivatives[i+1][b] / (derivatives[i][b] + epsilon)
    
    Raises:
        ValueError: If fewer than 2 derivatives provided
    """
    if len(derivatives) < 2:
        raise ValueError(f"Need at least 2 derivatives to compute ratios, got {len(derivatives)}")
    
    batch_size = derivatives[0].shape[0]
    n_ratios = len(derivatives) - 1
    
    # Stack derivatives into [B, n_orders] tensor
    # derivatives_stack[i, j] = j-th order derivative for sample i
    derivatives_stack = torch.stack(derivatives, dim=1)  # [B, n_orders]
    
    # Compute ratios: d_{n+1} / d_n for each consecutive pair
    # Use epsilon for numerical stability
    epsilon = 1e-12
    ratios = []
    
    for i in range(n_ratios):
        d_n = derivatives_stack[:, i]  # [B] - n-th order derivative
        d_n_plus_1 = derivatives_stack[:, i + 1]  # [B] - (n+1)-th order derivative
        
        # Compute ratio with numerical stability
        ratio = d_n_plus_1 / (d_n + epsilon)  # [B]
        ratios.append(ratio)
    
    # Stack into [B, n_ratios] tensor
    return torch.stack(ratios, dim=1)  # [B, n_ratios]


def estimate_lambda_from_derivatives(
    derivatives: List[torch.Tensor],
    abs_derivatives: bool = True,
    min_first_derivative: float = 1e-8
) -> Optional[float]:
    """
    Estimate λ from pre-computed derivatives.
        
    Args:
        derivatives: List of [B] tensors, derivatives[i] is (i+1)-th order derivative
        abs_derivatives: Whether to use absolute values of derivatives before computing ratios
        min_first_derivative: Minimum value of first derivative to consider valid (default: 1e-8)
    
    Returns:
        Mean λ value across valid samples, or None if computation fails
    
    Example:
        derivatives = [d1, d2, d3, d4]  # Each is [B]
        lambda_val = estimate_lambda_from_derivatives(derivatives)
    """
    if len(derivatives) < 2:
        return None
    
    # Optionally take absolute values
    if abs_derivatives:
        derivatives = [d.abs() for d in derivatives]
    
    # Compute ratios
    try:
        ratios = compute_derivative_ratios(derivatives)  # [B, n_ratios]
    except ValueError:
        return None
    
    batch_size = ratios.shape[0]
    n_ratios = ratios.shape[1]
    
    # Compute λ for each sample: mean(log(ratios)) across all ratio pairs
    lambda_values = []
    
    # Detach and move to CPU for numpy conversion
    ratios_cpu = ratios.detach().cpu()
    first_derivatives_cpu = derivatives[0].detach().cpu()
    
    for sample_idx in range(batch_size):
        # Get first derivative for this sample to check validity
        first_deriv = first_derivatives_cpu[sample_idx].item()
        
        if first_deriv < min_first_derivative:
            continue  # Skip samples with very small first derivative
        
        # Get ratios for this sample
        sample_ratios = ratios_cpu[sample_idx].numpy()  # [n_ratios]
        
        # Ensure ratios are positive before taking log (clamp to minimum positive value)
        # This prevents log of zero or negative numbers
        sample_ratios_clamped = np.maximum(sample_ratios, 1e-10)
        
        # Compute log(ratios) with numerical stability
        log_ratios = np.log(sample_ratios_clamped)
        
        # λ is the mean of log(ratios) for this sample
        lambda_val = np.mean(log_ratios)
        
        # Check for invalid values
        if np.isfinite(lambda_val):
            lambda_values.append(lambda_val)
    
    if len(lambda_values) == 0:
        return None
    
    return float(np.mean(lambda_values))


def estimate_lambda_per_direction(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: Callable,
    max_order: int = 4,
    K_dirs: int = 5,
    min_first_derivative: float = 1e-8,
    return_derivatives: bool = False,
) -> dict:
    """
    Estimate λ by computing it separately for each random direction, then aggregating.
    
    This provides the distribution of λ estimates
    across different directions, including mean and variance.
    
    Args:
        model: Neural network model
        inputs: Input batch [B, ...]
        labels: Labels [B]
        loss_fn: Loss function with reduction='none'
        max_order: Maximum derivative order (default: 4)
        K_dirs: Number of random directions (default: 5)
        min_first_derivative: Minimum first derivative threshold (default: 1e-8)
        return_derivatives: If True, also return all derivatives per direction
    
    Returns:
        Dictionary containing:
            - 'lambda_mean': Mean λ across directions
            - 'lambda_std': Standard deviation of λ
            - 'lambda_values': List of λ values (one per direction)
            - 'n_valid_directions': Number of directions that yielded valid λ
            - 'derivatives_per_dir': List of derivative lists (only if return_derivatives=True)

    """
    # Save training state and switch to eval mode
    was_training = model.training
    model.eval()
    
    batch_size = inputs.shape[0]
    input_shape = inputs.shape[1:]
    device = inputs.device
    
    lambda_values_per_direction = []
    derivatives_per_direction = []
    
    # Compute λ for each direction independently
    for dir_idx in range(K_dirs):
        # Sample random direction
        direction = sample_unit_directions(batch_size, input_shape, device)
        
        # Compute derivatives for this direction
        try:
            # create_graph=True is REQUIRED for max_order > 1
            # Higher-order derivatives need the computational graph to backprop through previous derivatives
            derivatives = compute_directional_derivatives(
                model=model,
                inputs=inputs,
                labels=labels,
                directions=direction,
                loss_fn=loss_fn,
                min_order=1,
                max_order=max_order,
                create_graph=True
            )
            # Take absolute values
            derivatives = [d.abs() for d in derivatives]
            
            # Compute log derivatives: log|d_n| for each order
            # Average across batch dimension
            log_derivatives = []
            valid_samples = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            for n, d_n in enumerate(derivatives):
                # Filter out samples with tiny first derivative
                if n == 0:
                    valid_samples &= (d_n >= min_first_derivative)
                
                # Filter out invalid values
                valid_samples &= torch.isfinite(d_n)
                valid_samples &= (d_n > 1e-10)
                
                if valid_samples.sum() == 0:
                    break
                
                # Compute mean log derivative across valid samples
                log_d_n = torch.log(d_n[valid_samples]).mean().item()
                log_derivatives.append(log_d_n)
            # Fit linear regression: log|d_n| = intercept + slope * n
            # Slope is our λ estimate for this direction
            if len(log_derivatives) >= 2:
                orders = np.arange(1, len(log_derivatives) + 1)
                log_derivs_array = np.array(log_derivatives)
                
                # Simple linear fit
                slope, intercept = np.polyfit(orders, log_derivs_array, 1)
                
                
                if np.isfinite(slope):
                    lambda_values_per_direction.append(slope)
                    
                    # Store derivatives if requested
                    if return_derivatives:
                        # Convert derivatives to list of floats for storage
                        deriv_list = []
                        for d_n in derivatives:
                            if valid_samples.sum() > 0:
                                deriv_list.append(d_n[valid_samples].mean().item())
                            else:
                                deriv_list.append(float('nan'))
                        derivatives_per_direction.append(deriv_list)

        except Exception as e:
            # Skip this direction if computation fails
            print(f"  Exception: {type(e).__name__}: {e}")
            continue
    
    # Restore training state
    if was_training:
        model.train()
    
    # Aggregate results
    if len(lambda_values_per_direction) == 0:
        result = {
            'lambda_mean': None,
            'lambda_std': None,
            'lambda_values': [],
            'n_valid_directions': 0
        }
        if return_derivatives:
            result['derivatives_per_dir'] = []
        return result
    
    lambda_values_array = np.array(lambda_values_per_direction)
    
    result = {
        'lambda_mean': float(np.mean(lambda_values_array)),
        'lambda_std': float(np.std(lambda_values_array, ddof=1 if len(lambda_values_array) > 1 else 0)),
        'lambda_values': lambda_values_per_direction,
        'n_valid_directions': len(lambda_values_per_direction)
    }
    
    if return_derivatives:
        result['derivatives_per_dir'] = derivatives_per_direction
    
    return result