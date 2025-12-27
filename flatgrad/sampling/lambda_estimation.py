'''
Estimate Lambda:
Calculated as \lambda = log(\frac{d_{n+1}}{d_n}) where d_n is the n-th order derivative
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