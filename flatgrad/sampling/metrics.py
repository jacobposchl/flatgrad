'''
Analysis and Metrics for Derivative Dynamics:
- Weighted Least Squares (WLS) fitting
- Analytic radius estimation
- Spectral edge estimation
- Cyclic pattern detection
'''

import numpy as np
import math
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from scipy import stats


@dataclass
class WLSResult:
    """
    Result of weighted least squares fitting.
    
    Attributes:
        slope: Fitted slope coefficient
        intercept: Fitted intercept coefficient
        slope_std: Standard error of slope estimate
        intercept_std: Standard error of intercept estimate
        slope_ci_lower: Lower bound of 95% confidence interval for slope
        slope_ci_upper: Upper bound of 95% confidence interval for slope
        intercept_ci_lower: Lower bound of 95% confidence interval for intercept
        intercept_ci_upper: Upper bound of 95% confidence interval for intercept
        r_squared: R-squared goodness of fit
        p_value: p-value for slope significance test (two-tailed)
        residuals: Residuals from the fit
        degrees_of_freedom: Degrees of freedom for the fit
    """
    slope: float
    intercept: float
    slope_std: float
    intercept_std: float
    slope_ci_lower: float
    slope_ci_upper: float
    intercept_ci_lower: float
    intercept_ci_upper: float
    r_squared: float
    p_value: float
    residuals: np.ndarray
    degrees_of_freedom: int


def wls_linear(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    confidence_level: float = 0.95
) -> WLSResult:
    """
    Perform weighted least squares linear fit: y = slope * x + intercept.
    
    Args:
        x: Independent variable [n]
        y: Dependent variable [n]
        weights: Optional weights [n]. If None, uses uniform weights (equivalent to OLS).
                Weights are typically inverse variance or based on data reliability.
        confidence_level: Confidence level for intervals (default: 0.95)
    
    Returns:
        WLSResult containing fitted parameters, standard errors, confidence intervals,
        R-squared, p-value, and residuals.
    
    Raises:
        ValueError: If x and y have different lengths, or if weights have wrong shape
    
    Example:
        >>> x = np.array([1, 2, 3, 4])
        >>> y = np.array([2.1, 3.9, 6.2, 8.1])
        >>> result = wls_linear(x, y)
        >>> print(f"Slope: {result.slope:.3f} ± {result.slope_std:.3f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    if len(x) < 2:
        raise ValueError(f"Need at least 2 data points, got {len(x)}")
    
    # Set up weights
    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = np.asarray(weights)
        if weights.shape != x.shape:
            raise ValueError(f"weights must have same shape as x, got {weights.shape} vs {x.shape}")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
    
    # Normalize weights (doesn't affect fit, but helps with numerical stability)
    weights = weights / (np.sum(weights) / len(weights))
    
    # Weighted least squares formula
    # Solve: minimize sum(w_i * (y_i - (a*x_i + b))^2)
    # Closed form solution:
    sum_w = np.sum(weights)
    sum_wx = np.sum(weights * x)
    sum_wy = np.sum(weights * y)
    sum_wxy = np.sum(weights * x * y)
    sum_wx2 = np.sum(weights * x * x)
    
    # Denominator
    denom = sum_w * sum_wx2 - sum_wx * sum_wx
    
    if abs(denom) < 1e-12:
        raise ValueError("Singular matrix: x values are all the same or weights are invalid")
    
    # Compute slope and intercept
    slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
    intercept = (sum_wx2 * sum_wy - sum_wx * sum_wxy) / denom
    
    # Compute fitted values and residuals
    y_fitted = slope * x + intercept
    residuals = y - y_fitted
    weighted_residuals = weights * residuals
    
    # Degrees of freedom
    dof = len(x) - 2
    
    # Compute variance of residuals (weighted mean squared error)
    mse = np.sum(weighted_residuals ** 2) / dof if dof > 0 else 0.0
    
    # Variance-covariance matrix
    # Var(slope) = mse * sum(w) / denom
    # Var(intercept) = mse * sum(w*x^2) / denom
    # Cov(slope, intercept) = -mse * sum(w*x) / denom
    slope_var = mse * sum_w / denom
    intercept_var = mse * sum_wx2 / denom
    
    slope_std = np.sqrt(slope_var) if slope_var > 0 else 0.0
    intercept_std = np.sqrt(intercept_var) if intercept_var > 0 else 0.0
    
    # Confidence intervals (using t-distribution)
    t_critical = stats.t.ppf((1 + confidence_level) / 2, dof) if dof > 0 else 1.96
    
    slope_ci_lower = slope - t_critical * slope_std
    slope_ci_upper = slope + t_critical * slope_std
    intercept_ci_lower = intercept - t_critical * intercept_std
    intercept_ci_upper = intercept + t_critical * intercept_std
    
    # R-squared
    y_mean = np.sum(weights * y) / sum_w
    ss_total = np.sum(weights * (y - y_mean) ** 2)
    ss_residual = np.sum(weighted_residuals ** 2)
    r_squared = 1.0 - (ss_residual / ss_total) if ss_total > 1e-12 else 0.0
    
    # p-value for slope (two-tailed t-test: H0: slope = 0)
    if slope_std > 1e-12:
        t_statistic = slope / slope_std
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), dof)) if dof > 0 else 1.0
    else:
        p_value = 0.0 if abs(slope) > 1e-12 else 1.0
    
    return WLSResult(
        slope=slope,
        intercept=intercept,
        slope_std=slope_std,
        intercept_std=intercept_std,
        slope_ci_lower=slope_ci_lower,
        slope_ci_upper=slope_ci_upper,
        intercept_ci_lower=intercept_ci_lower,
        intercept_ci_upper=intercept_ci_upper,
        r_squared=r_squared,
        p_value=p_value,
        residuals=residuals,
        degrees_of_freedom=dof
    )


def wls_linear_log(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
    epsilon: float = 1e-12
) -> WLSResult:
    """
    Perform weighted least squares fit with logarithmic term: y = slope * log(x + epsilon) + intercept.
    
    Useful for fitting exponential decay patterns where y ~ log-scale relationship.
    
    Args:
        x: Independent variable [n] (will be transformed to log space)
        y: Dependent variable [n]
        weights: Optional weights [n]. If None, uses uniform weights.
        confidence_level: Confidence level for intervals (default: 0.95)
        epsilon: Small value added before taking log for numerical stability (default: 1e-12)
    
    Returns:
        WLSResult containing fitted parameters and statistics.
        Note: The slope and intercept are for the log-transformed model.
    
    Example:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2.0, 1.5, 1.2, 1.0, 0.9])  # Decaying pattern
        >>> result = wls_linear_log(x, y)
        >>> print(f"Decay rate: {result.slope:.3f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if np.any(x <= -epsilon):
        raise ValueError(f"x values must be > -epsilon ({epsilon}), got min(x) = {np.min(x)}")
    
    # Transform x to log space
    x_log = np.log(x + epsilon)
    
    # Perform WLS on log-transformed data
    return wls_linear(x_log, y, weights=weights, confidence_level=confidence_level)


def compute_analytic_radius(
    derivatives: List[torch.Tensor],
    abs_derivatives: bool = True,
    min_order: int = 1,
    confidence_level: float = 0.95
) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate analytic radius R from factorial-normalized Taylor coefficients.
    
    The analytic radius R is the radius of convergence of the Taylor series expansion.
    If derivatives follow d_n ~ R^(-n) * n!, then R can be estimated from the decay rate.
    
    Mathematical approach:
    - Normalize by factorials: a_n = |d_n| / n!
    - Fit exponential decay: log(a_n) = -n * log(R) + constant
    - Extract R from the slope
    
    Args:
        derivatives: List of [B] tensors, derivatives[i] is the (i+1)-th order derivative
        abs_derivatives: Whether to use absolute values of derivatives (default: True)
        min_order: Minimum order to include in fit (default: 1)
        confidence_level: Confidence level for intervals (default: 0.95)
    
    Returns:
        If batch_size == 1: scalar R estimate
        If batch_size > 1: tuple (R_values [B], R_confidence [B])
        Returns None if computation fails
    
    Example:
        >>> derivatives = [d1, d2, d3, d4]  # Each is [B]
        >>> R = compute_analytic_radius(derivatives)
    """
    if len(derivatives) < 2:
        return None
    
    # Convert to numpy and handle absolute values
    derivatives_np = []
    for d in derivatives:
        d_cpu = d.detach().cpu()
        if abs_derivatives:
            d_cpu = d_cpu.abs()
        derivatives_np.append(d_cpu.numpy())
    
    batch_size = derivatives_np[0].shape[0]
    n_orders = len(derivatives_np)
    
    # Compute factorial-normalized coefficients for each sample
    orders = np.arange(min_order, n_orders + min_order)  # [1, 2, 3, ...]
    factorials = np.array([math.factorial(n) for n in orders])
    
    R_values = []
    R_confidences = []
    
    for sample_idx in range(batch_size):
        # Extract derivative sequence for this sample
        deriv_sequence = np.array([d[sample_idx] for d in derivatives_np])
        
        # Skip if all zeros
        if np.all(deriv_sequence < 1e-12):
            R_values.append(np.nan)
            R_confidences.append(np.nan)
            continue
        
        # Normalize by factorials: a_n = |d_n| / n!
        a_n = deriv_sequence / factorials
        
        # Skip if any values are invalid
        if np.any(a_n < 0) or np.any(~np.isfinite(a_n)):
            R_values.append(np.nan)
            R_confidences.append(np.nan)
            continue
        
        # Fit log(a_n) vs n: log(a_n) = -n * log(R) + constant
        # So slope = -log(R), therefore R = exp(-slope)
        try:
            log_a_n = np.log(a_n + 1e-12)
            
            # Use WLS to fit log(a_n) = slope * n + intercept
            # Weights: use inverse of variance or just uniform for now
            result = wls_linear(orders, log_a_n, confidence_level=confidence_level)
            
            # Extract R from slope: R = exp(-slope)
            # But note: if log(a_n) = -n * log(R) + c, then slope = -log(R)
            R_estimate = np.exp(-result.slope) if result.slope < 0 else np.inf
            
            # Handle edge cases
            if not np.isfinite(R_estimate) or R_estimate <= 0:
                R_values.append(np.nan)
                R_confidences.append(np.nan)
            else:
                R_values.append(R_estimate)
                # Use confidence interval for slope to compute CI for R
                R_lower = np.exp(-result.slope_ci_upper) if result.slope_ci_upper < 0 else np.inf
                R_upper = np.exp(-result.slope_ci_lower) if result.slope_ci_lower < 0 else 0.0
                R_confidences.append((R_lower, R_upper))
                
        except (ValueError, RuntimeError):
            R_values.append(np.nan)
            R_confidences.append(np.nan)
    
    # Return format depends on batch size
    R_values = np.array(R_values)
    
    if batch_size == 1:
        return float(R_values[0]) if np.isfinite(R_values[0]) else None
    else:
        R_conf_array = np.array(R_confidences)
        return R_values, R_conf_array


def compute_spectral_edge(
    derivatives: List[torch.Tensor],
    normalize_by_factorial: bool = False,
    confidence_level: float = 0.95
) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate spectral edge Ω from L2 norms of derivatives.
    
    If derivatives follow ||d_n||₂ ~ Ω^n, then log(||d_n||₂) = n * log(Ω) + constant.
    The spectral edge Ω characterizes the frequency content of the loss landscape.
    
    Args:
        derivatives: List of derivative tensors. Can be:
                    - List of [B] tensors (directional derivatives), or
                    - List of [B, ...] tensors (full gradients)
        normalize_by_factorial: Whether to normalize by n! before computing norms (default: False)
        confidence_level: Confidence level for intervals (default: 0.95)
    
    Returns:
        If batch_size == 1: scalar Ω estimate
        If batch_size > 1: tuple (omega_values [B], omega_confidence [B])
        Returns None if computation fails
    
    Example:
        >>> derivatives = [grad1, grad2, grad3]  # Full gradients or directional derivatives
        >>> omega = compute_spectral_edge(derivatives)
    """
    if len(derivatives) < 2:
        return None
    
    # Compute L2 norms for each order
    batch_size = derivatives[0].shape[0]
    n_orders = len(derivatives)
    orders = np.arange(1, n_orders + 1)
    
    # Compute norms
    norms = []
    for i, d in enumerate(derivatives):
        d_cpu = d.detach().cpu()
        
        # Flatten to compute L2 norm
        d_flat = d_cpu.view(batch_size, -1)  # [B, D]
        
        # Compute L2 norm per sample
        norm_per_sample = d_flat.norm(dim=1).numpy()  # [B]
        
        # Optionally normalize by factorial
        if normalize_by_factorial:
            factorial = math.factorial(i + 1)
            norm_per_sample = norm_per_sample / factorial
        
        norms.append(norm_per_sample)
    
    # Stack: [n_orders, batch_size]
    norms_array = np.stack(norms, axis=0)
    
    omega_values = []
    omega_confidences = []
    
    for sample_idx in range(batch_size):
        # Extract norm sequence for this sample
        norm_sequence = norms_array[:, sample_idx]
        
        # Skip if all zeros or invalid
        if np.all(norm_sequence < 1e-12) or np.any(~np.isfinite(norm_sequence)):
            omega_values.append(np.nan)
            omega_confidences.append(np.nan)
            continue
        
        # Fit log(||d_n||₂) = n * log(Ω) + constant
        try:
            log_norms = np.log(norm_sequence + 1e-12)
            
            # Fit: log(||d_n||₂) = slope * n + intercept
            # So slope = log(Ω), therefore Ω = exp(slope)
            result = wls_linear(orders, log_norms, confidence_level=confidence_level)
            
            # Extract Ω from slope
            omega_estimate = np.exp(result.slope)
            
            if not np.isfinite(omega_estimate) or omega_estimate <= 0:
                omega_values.append(np.nan)
                omega_confidences.append(np.nan)
            else:
                omega_values.append(omega_estimate)
                # Confidence interval for Ω
                omega_lower = np.exp(result.slope_ci_lower) if result.slope_ci_lower > -10 else 0.0
                omega_upper = np.exp(result.slope_ci_upper) if result.slope_ci_upper < 10 else np.inf
                omega_confidences.append((omega_lower, omega_upper))
                
        except (ValueError, RuntimeError):
            omega_values.append(np.nan)
            omega_confidences.append(np.nan)
    
    # Return format depends on batch size
    omega_values = np.array(omega_values)
    
    if batch_size == 1:
        return float(omega_values[0]) if np.isfinite(omega_values[0]) else None
    else:
        omega_conf_array = np.array(omega_confidences)
        return omega_values, omega_conf_array


def detect_cyclic(
    derivatives: List[torch.Tensor],
    threshold: float = 0.7,
    abs_derivatives: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect cyclic patterns (e.g., sin/cos four-cycle) in derivative sequences.
    
    Looks for patterns like [A, B, -A, -B, A, ...] or [A, 0, -A, 0, A, ...]
    that indicate oscillatory behavior in the loss landscape.
    
    Args:
        derivatives: List of [B] tensors, derivatives[i] is the (i+1)-th order derivative
        threshold: Correlation threshold for detecting cyclic pattern (default: 0.7)
        abs_derivatives: Whether to use absolute values before pattern matching (default: True)
    
    Returns:
        Tuple of (is_cyclic [B], cycle_period [B], correlation_strength [B])
        - is_cyclic: Boolean array indicating if pattern is cyclic
        - cycle_period: Detected period (typically 2 or 4 for sin/cos patterns)
        - correlation_strength: Correlation with ideal cyclic pattern (0-1)
    
    Example:
        >>> derivatives = [d1, d2, d3, d4, d5, d6, d7, d8]
        >>> is_cyclic, period, corr = detect_cyclic(derivatives)
    """
    if len(derivatives) < 4:
        # Need at least 4 points to detect 4-cycle
        batch_size = derivatives[0].shape[0] if len(derivatives) > 0 else 0
        return (
            np.zeros(batch_size, dtype=bool),
            np.full(batch_size, np.nan),
            np.zeros(batch_size)
        )
    
    # Convert to numpy
    derivatives_np = []
    for d in derivatives:
        d_cpu = d.detach().cpu()
        if abs_derivatives:
            d_cpu = d_cpu.abs()
        derivatives_np.append(d_cpu.numpy())
    
    batch_size = derivatives_np[0].shape[0]
    n_orders = len(derivatives_np)
    
    # Ideal cyclic patterns to match
    # Pattern 1: [1, 0, 1, 0, 1, 0, ...] (2-cycle)
    # Pattern 2: [1, 1, 1, 1, 1, 1, ...] then [1, -1, 1, -1, ...] (sign alternation)
    # Pattern 3: [1, 0, -1, 0, 1, 0, ...] (4-cycle)
    # Pattern 4: [0, 1, 0, -1, 0, 1, ...] (4-cycle, shifted)
    # Pattern 5: [1, 1, -1, -1, 1, 1, ...] (4-cycle)
    
    ideal_patterns = [
        (np.array([1, 0] * (n_orders // 2 + 1))[:n_orders], 2),  # 2-cycle
        (np.array([1, 0, -1, 0] * (n_orders // 4 + 1))[:n_orders], 4),  # 4-cycle type 1
        (np.array([0, 1, 0, -1] * (n_orders // 4 + 1))[:n_orders], 4),  # 4-cycle type 2
        (np.array([1, 1, -1, -1] * (n_orders // 4 + 1))[:n_orders], 4),  # 4-cycle type 3
    ]
    
    is_cyclic = np.zeros(batch_size, dtype=bool)
    cycle_period = np.full(batch_size, np.nan)
    correlation_strength = np.zeros(batch_size)
    
    for sample_idx in range(batch_size):
        # Extract derivative sequence
        deriv_sequence = np.array([np.asarray(d[sample_idx]).flatten()[0] for d in derivatives_np])
        
        # Ensure 1D array
        deriv_sequence = deriv_sequence.flatten()
        
        # Normalize (zero-mean, unit-variance) for correlation
        if np.std(deriv_sequence) < 1e-12:
            continue  # Constant sequence, not cyclic
        
        deriv_normalized = (deriv_sequence - np.mean(deriv_sequence)) / (np.std(deriv_sequence) + 1e-12)
        deriv_normalized = deriv_normalized.flatten()  # Ensure 1D
        
        # Try each ideal pattern
        best_correlation = -1.0
        best_period = np.nan
        
        for pattern, period in ideal_patterns:
            # Ensure pattern is 1D and has same length as deriv_sequence
            pattern_flat = pattern[:len(deriv_normalized)].flatten()
            
            # Skip if lengths don't match
            if len(pattern_flat) != len(deriv_normalized):
                continue
            
            # Normalize pattern
            if np.std(pattern_flat) < 1e-12:
                continue  # Skip constant patterns
            
            pattern_normalized = (pattern_flat - np.mean(pattern_flat)) / (np.std(pattern_flat) + 1e-12)
            pattern_normalized = pattern_normalized.flatten()  # Ensure 1D
            
            # Compute correlation (both should be 1D arrays of same length)
            try:
                correlation_matrix = np.corrcoef(deriv_normalized, pattern_normalized)
                correlation = correlation_matrix[0, 1]
            except (ValueError, IndexError):
                continue
            
            if np.isfinite(correlation) and correlation > best_correlation:
                best_correlation = correlation
                best_period = period
        
        # Check if correlation exceeds threshold
        if best_correlation >= threshold:
            is_cyclic[sample_idx] = True
            cycle_period[sample_idx] = best_period
            correlation_strength[sample_idx] = best_correlation
        else:
            correlation_strength[sample_idx] = best_correlation
    
    return is_cyclic, cycle_period, correlation_strength

