'''
Test Models with Known Analytical Properties:

This module provides test models for validating lambda estimation and metrics.
Models are designed to have known analytical properties (lambda, R, omega) that
can be compared against estimated values.

All models are data-agnostic and work with arbitrary input shapes.
'''

import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Optional, Tuple, Union


class PolynomialModel(nn.Module):
    """
    Polynomial model: f(x) = Σ a_i * x^i for i = 0 to degree
    
    Analytical Properties:
    - N-th derivative is exactly N! * a_N (constant for order N, zero for orders > N)
    - For degree N polynomial, all derivatives of order > N are zero
    - Can compute exact lambda from polynomial structure
    
    The model computes: output = Σ (coeff[i] * x^i) for i in [0, degree]
    where x is the input flattened along spatial dimensions.
    
    Args:
        degree: Maximum polynomial degree (0 = constant, 1 = linear, 2 = quadratic, etc.)
        input_dim: Input dimension (flattened size)
        output_dim: Output dimension (default: 1 for regression, or num_classes for classification)
        coefficients: Optional tensor of shape [degree+1, output_dim, input_dim] for each power
                     If None, initializes with random values
        mode: 'regression' (scalar output) or 'classification' (logits output)
    """
    
    def __init__(
        self,
        degree: int,
        input_dim: int,
        output_dim: int = 1,
        coefficients: Optional[torch.Tensor] = None,
        mode: str = 'regression'
    ):
        super().__init__()
        
        if degree < 0:
            raise ValueError(f"degree must be >= 0, got {degree}")
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        
        self.degree = degree
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        
        # Coefficients: shape [degree+1, output_dim, input_dim]
        # coeff[i] gives coefficients for x^i term
        if coefficients is None:
            # Initialize with small random values
            coefficients = torch.randn(degree + 1, output_dim, input_dim) * 0.1
        
        if coefficients.shape != (degree + 1, output_dim, input_dim):
            raise ValueError(
                f"coefficients must have shape {(degree + 1, output_dim, input_dim)}, "
                f"got {coefficients.shape}"
            )
        
        self.coefficients = nn.Parameter(coefficients)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: computes polynomial expansion.
        
        Args:
            x: Input tensor [B, ...] - any shape, will be flattened
        
        Returns:
            Output tensor [B, output_dim]
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # [B, input_dim]
        
        if x_flat.shape[1] != self.input_dim:
            raise ValueError(
                f"Flattened input dimension {x_flat.shape[1]} doesn't match "
                f"model input_dim {self.input_dim}"
            )
        
        # Compute polynomial: f_k(x) = Σ_i Σ_d (coeff[i, k, d] * x_d^i)
        # For each output dimension k:
        #   - Constant term (i=0): sum of constant coefficients
        #   - Linear term (i=1): weighted sum of x
        #   - Higher order (i>1): weighted sum of x^i
        
        output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        # Constant term (i=0)
        # coeff[0] is [output_dim, input_dim] - for constant term, we sum over input_dim
        constant = self.coefficients[0].sum(dim=1)  # [output_dim]
        output = output + constant.unsqueeze(0).expand(batch_size, -1)  # [batch_size, output_dim]
        
        # Higher order terms
        for i in range(1, self.degree + 1):
            x_power = x_flat.pow(i)  # [B, input_dim]
            
            # For each output dimension: weighted sum of x_power
            # coeff[i, k, :] @ x_power[b, :] for each batch b and output k
            # This is: coeff[i] @ x_power.T -> [output_dim, B], then transpose
            output = output + (self.coefficients[i] @ x_power.T).T  # [B, output_dim]
        
        return output


class QuadraticModel(PolynomialModel):
    """
    Convenience class for quadratic (degree-2) polynomial models.
    
    Analytical Properties:
    - First derivative: linear function of x
    - Second derivative: constant (2 * quadratic_coefficient)
    - Third and higher derivatives: exactly zero
    - Lambda: can be computed exactly for this structure
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        coefficients: Optional[torch.Tensor] = None,
        mode: str = 'regression'
    ):
        super().__init__(
            degree=2,
            input_dim=input_dim,
            output_dim=output_dim,
            coefficients=coefficients,
            mode=mode
        )


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron with configurable architecture.
    
    Standard neural network model for general testing.
    No known analytical properties, but useful for general validation.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions, e.g., [64, 32] for two hidden layers
        output_dim: Output dimension
        activation: Activation function name: 'tanh', 'relu', 'gelu', 'sigmoid'
        use_batch_norm: Whether to use batch normalization (default: False)
        dropout: Dropout probability (default: 0.0, no dropout)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'tanh',
        use_batch_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        if any(h < 1 for h in hidden_dims):
            raise ValueError(f"All hidden_dims must be >= 1, got {hidden_dims}")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_name = activation
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add activation (except after last layer)
            if i < len(dims) - 2:
                if activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                
                # Optional batch norm
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                
                # Optional dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            x: Input tensor [B, ...] - any shape, will be flattened
        
        Returns:
            Output tensor [B, output_dim]
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # [B, input_dim]
        
        if x_flat.shape[1] != self.input_dim:
            raise ValueError(
                f"Flattened input dimension {x_flat.shape[1]} doesn't match "
                f"model input_dim {self.input_dim}"
            )
        
        return self.network(x_flat)


class ExponentialDecayModel(nn.Module):
    """
    Model with exponentially decaying derivative magnitudes.
    
    Analytical Properties:
    - Uses exp(w·x) which produces true exponential decay in directional derivatives
    - For f(x) = exp(w·x), directional derivatives are: D^n f = (w·u)^n * exp(w·x)
    - When combined with MSE loss, derivatives decay approximately exponentially
    - lambda ≈ log(|w·u|) where w is the weight vector and u is the direction
    - R ≈ 1 / |w·u|
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension (default: 1)
        decay_factor: Target decay rate for derivatives (default: 0.5)
                     This sets |w| such that derivatives decay by this factor
        base_scale: Base scale for output (default: 1.0)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        decay_factor: float = 0.5,
        base_scale: float = 1.0
    ):
        super().__init__()
        
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        if decay_factor <= 0 or decay_factor >= 1:
            raise ValueError(f"decay_factor must be in (0, 1), got {decay_factor}")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.decay_factor = decay_factor
        self.base_scale = base_scale
        self.analytical_lambda = math.log(decay_factor)
        self.analytical_radius = 1.0 / decay_factor
        
        # Fixed weight vector - the magnitude controls the decay rate
        # For exp(w·x), directional derivative D^n[exp(w·x)] = (w·u)^n * exp(w·x)
        # We want |w·u| ≈ decay_factor on average, so we scale w accordingly
        # Setting |w| = decay_factor * sqrt(input_dim) makes |w·u| ≈ decay_factor for unit u
        weights = torch.randn(output_dim, input_dim)
        weights = weights / (weights.norm(dim=1, keepdim=True) + 1e-8)
        weights = weights * decay_factor * math.sqrt(input_dim)
        # Use Parameter so gradients flow, but we won't train it
        self.weights = nn.Parameter(weights, requires_grad=False)
        
        # Bias and scale
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.scale = nn.Parameter(torch.ones(output_dim) * base_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output = scale * exp(weights·x + bias)
        
        The exponential ensures true exponential behavior in derivatives.
        
        Args:
            x: Input tensor [B, ...] - any shape, will be flattened
        
        Returns:
            Output tensor [B, output_dim]
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # [B, input_dim]
        
        if x_flat.shape[1] != self.input_dim:
            raise ValueError(
                f"Flattened input dimension {x_flat.shape[1]} doesn't match "
                f"model input_dim {self.input_dim}"
            )
        
        # Compute linear transformation: w·x + b
        z = torch.matmul(x_flat, self.weights.T) + self.bias  # [B, output_dim]
        
        # Apply exponential with scaling
        # Clamp to avoid numerical overflow
        z = torch.clamp(z, min=-10, max=10)
        output = self.scale * torch.exp(z)
        
        return output


class SinusoidalModel(nn.Module):
    """
    Model with sin/cos components that produce cyclic derivative patterns.
    
    Analytical Properties:
    - Produces 4-cycle pattern in derivatives: [A, B, -A, -B, A, ...]
    - Should be detected by detect_cyclic() function
    - Periodicity depends on frequency parameter
    
    The model uses sin/cos functions to create oscillatory behavior.
    The key is to compute f(x) = sin(w·x) where w is a fixed weight vector.
    Then directional derivatives D^n[f](x; u) will follow the pattern:
    D^1 = w·u * cos(w·x)
    D^2 = -(w·u)^2 * sin(w·x)
    D^3 = -(w·u)^3 * cos(w·x)
    D^4 = (w·u)^4 * sin(w·x)
    showing a 4-cycle in the sin/cos pattern.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension (default: 1)
        frequency: Frequency of oscillation (default: 1.0)
                  Higher frequency = faster oscillation
        amplitude: Amplitude of oscillation (default: 1.0)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        frequency: float = 1.0,
        amplitude: float = 1.0
    ):
        super().__init__()
        
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.frequency = frequency
        self.amplitude = amplitude
        
        # Fixed weight vector for each output dimension
        # Normalize to unit length for stable cyclic patterns, then scale by frequency
        # Use Parameter(requires_grad=False) instead of buffer for gradient flow
        weights = torch.randn(output_dim, input_dim)
        weights = weights / (weights.norm(dim=1, keepdim=True) + 1e-8)  # Unit normalize
        weights = weights * frequency * math.sqrt(input_dim)  # Scale appropriately
        self.weights = nn.Parameter(weights, requires_grad=False)
        
        # Bias and scale
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.scale = nn.Parameter(torch.ones(output_dim) * amplitude)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output = scale * sin(weights·x + bias)
        
        Using sin ensures cyclic pattern in directional derivatives.
        
        Args:
            x: Input tensor [B, ...] - any shape, will be flattened
        
        Returns:
            Output tensor [B, output_dim]
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # [B, input_dim]
        
        if x_flat.shape[1] != self.input_dim:
            raise ValueError(
                f"Flattened input dimension {x_flat.shape[1]} doesn't match "
                f"model input_dim {self.input_dim}"
            )
        
        # Compute linear transformation: w·x
        z = torch.matmul(x_flat, self.weights.T) + self.bias  # [B, output_dim]
        
        # Apply sin with scaling
        output = self.scale * torch.sin(z)
        
        return output


class LinearCombinationModel(nn.Module):
    """
    Model that's a linear combination of basis models with known properties.
    
    Composes multiple sub-models (polynomial + sinusoidal + exponential) to create
    complex behaviors. Can analytically compute overall properties from components.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension (default: 1)
        components: List of (model, weight) tuples to combine
                   If None, creates default combination
        weights: Optional weights for combining components (default: equal weights)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        components: Optional[List[Tuple[nn.Module, float]]] = None,
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if components is None:
            # Create default combination: polynomial + sinusoidal + exponential
            poly = PolynomialModel(degree=2, input_dim=input_dim, output_dim=output_dim)
            sin = SinusoidalModel(input_dim=input_dim, output_dim=output_dim, frequency=1.0)
            exp = ExponentialDecayModel(input_dim=input_dim, output_dim=output_dim, decay_factor=0.5)
            
            components = [
                (poly, 0.4),
                (sin, 0.3),
                (exp, 0.3)
            ]
        
        self.components = nn.ModuleList([comp for comp, _ in components])
        self.weights = nn.Parameter(torch.tensor([w for _, w in components], dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: weighted sum of component model outputs.
        
        Args:
            x: Input tensor [B, ...] - any shape
        
        Returns:
            Output tensor [B, output_dim]
        """
        # Compute outputs from each component
        outputs = []
        for component in self.components:
            outputs.append(component(x))
        
        # Weighted combination
        # weights is [num_components]
        # Stack outputs: [num_components, B, output_dim]
        stacked = torch.stack(outputs, dim=0)  # [num_components, B, output_dim]
        
        # Weighted sum: weights @ stacked -> [B, output_dim]
        # weights: [num_components], stacked: [num_components, B, output_dim]
        weighted = (self.weights.unsqueeze(1).unsqueeze(2) * stacked).sum(dim=0)
        
        return weighted


def create_test_model(
    model_type: str,
    input_dim: int,
    output_dim: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create test models easily.
    
    Args:
        model_type: Type of model to create:
                   - 'polynomial': PolynomialModel
                   - 'quadratic': QuadraticModel
                   - 'mlp': SimpleMLP
                   - 'exponential': ExponentialDecayModel
                   - 'sinusoidal': SinusoidalModel
                   - 'linear_combination': LinearCombinationModel
        input_dim: Input dimension
        output_dim: Output dimension (default: 1)
        **kwargs: Additional arguments specific to each model type
    
    Returns:
        Configured model instance
    
    Examples:
        >>> # Create a quadratic model
        >>> model = create_test_model('quadratic', input_dim=10, output_dim=1)
        
        >>> # Create an MLP with custom architecture
        >>> model = create_test_model('mlp', input_dim=10, output_dim=10,
        ...                          hidden_dims=[64, 32], activation='tanh')
        
        >>> # Create exponential decay model
        >>> model = create_test_model('exponential', input_dim=10, decay_factor=0.6)
    """
    model_type = model_type.lower()
    
    if model_type == 'polynomial':
        degree = kwargs.get('degree', 2)
        return PolynomialModel(
            degree=degree,
            input_dim=input_dim,
            output_dim=output_dim,
            coefficients=kwargs.get('coefficients', None),
            mode=kwargs.get('mode', 'regression')
        )
    
    elif model_type == 'quadratic':
        return QuadraticModel(
            input_dim=input_dim,
            output_dim=output_dim,
            coefficients=kwargs.get('coefficients', None),
            mode=kwargs.get('mode', 'regression')
        )
    
    elif model_type == 'mlp':
        hidden_dims = kwargs.get('hidden_dims', [32])
        activation = kwargs.get('activation', 'tanh')
        use_batch_norm = kwargs.get('use_batch_norm', False)
        dropout = kwargs.get('dropout', 0.0)
        return SimpleMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout
        )
    
    elif model_type == 'exponential':
        decay_factor = kwargs.get('decay_factor', 0.5)
        base_scale = kwargs.get('base_scale', 1.0)
        return ExponentialDecayModel(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_factor=decay_factor,
            base_scale=base_scale
        )
    
    elif model_type == 'sinusoidal':
        frequency = kwargs.get('frequency', 1.0)
        amplitude = kwargs.get('amplitude', 1.0)
        return SinusoidalModel(
            input_dim=input_dim,
            output_dim=output_dim,
            frequency=frequency,
            amplitude=amplitude
        )
    
    elif model_type == 'linear_combination':
        components = kwargs.get('components', None)
        weights = kwargs.get('weights', None)
        return LinearCombinationModel(
            input_dim=input_dim,
            output_dim=output_dim,
            components=components,
            weights=weights
        )
    
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Must be one of: 'polynomial', 'quadratic', 'mlp', 'exponential', "
            f"'sinusoidal', 'linear_combination'"
        )


def compute_analytical_lambda(
    model: nn.Module,
    inputs: torch.Tensor,
    directions: torch.Tensor
) -> Optional[float]:
    """
    Compute analytical lambda value for models that support it.
    
    Currently supports:
    - ExponentialDecayModel: returns model.analytical_lambda
    - Other models: returns None (no analytical solution available)
    
    Args:
        model: Model instance
        inputs: Input tensor [B, ...]
        directions: Direction tensor [B, ...]
    
    Returns:
        Analytical lambda value, or None if not available
    """
    if isinstance(model, ExponentialDecayModel):
        return model.analytical_lambda
    
    # For other models, analytical lambda may depend on inputs/directions
    # Could be extended in the future
    return None


def compute_analytical_radius(
    model: nn.Module,
    inputs: torch.Tensor,
    directions: torch.Tensor
) -> Optional[float]:
    """
    Compute analytical radius R for models that support it.
    
    Currently supports:
    - ExponentialDecayModel: returns model.analytical_radius
    
    Args:
        model: Model instance
        inputs: Input tensor [B, ...]
        directions: Direction tensor [B, ...]
    
    Returns:
        Analytical radius R, or None if not available
    """
    if isinstance(model, ExponentialDecayModel):
        return model.analytical_radius
    
    return None

