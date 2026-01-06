"""
Regularization methods for proof-of-concept experiments.

Implements standard regularization techniques:
- SAM (Sharpness-Aware Minimization)
- Input Gradient Penalty
- Label Smoothing
- Data Augmentation transforms
- Weight decay (via optimizer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Callable, Optional


class SAMOptimizer:
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.
    
    Implements the SAM optimization strategy from:
    "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    (Foret et al., 2021)
    
    SAM seeks parameters that lie in neighborhoods with uniformly low loss,
    improving generalization by finding flatter minima.
    
    Args:
        optimizer: Base optimizer (e.g., torch.optim.Adam)
        rho: Neighborhood size for sharpness measurement (default: 0.05)
    
    Example:
        >>> base_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> optimizer = SAMOptimizer(base_optimizer, rho=0.05)
        >>> 
        >>> # First forward-backward pass
        >>> loss = criterion(model(inputs), labels)
        >>> loss.backward()
        >>> optimizer.first_step(zero_grad=True)
        >>> 
        >>> # Second forward-backward pass
        >>> criterion(model(inputs), labels).backward()
        >>> optimizer.second_step(zero_grad=True)
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, rho: float = 0.05):
        self.optimizer = optimizer
        self.rho = rho
        self.state = {}
        
    def first_step(self, zero_grad: bool = False):
        """
        Perform first step: compute gradient and ascend to find worst-case perturbation.
        
        Args:
            zero_grad: Whether to zero gradients after the step
        """
        # Compute gradient norm
        grad_norm = self._grad_norm()
        
        # Save current parameters and compute epsilon (perturbation)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Save original parameters
                self.state[p] = p.data.clone()
                
                # Compute perturbation: epsilon = rho * grad / ||grad||
                epsilon = (self.rho / (grad_norm + 1e-12)) * p.grad
                
                # Ascend: p_adv = p + epsilon
                p.data.add_(epsilon)
        
        if zero_grad:
            self.zero_grad()
    
    def second_step(self, zero_grad: bool = False):
        """
        Perform second step: update parameters using gradient at perturbed point.
        
        Args:
            zero_grad: Whether to zero gradients after the step
        """
        # Restore original parameters and apply gradient update
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in self.state:
                    continue
                
                # Restore original parameter
                p.data = self.state[p]
        
        # Standard optimizer step using gradient at perturbed point
        self.optimizer.step()
        
        if zero_grad:
            self.zero_grad()
        
        # Clear saved state
        self.state = {}
    
    def step(self):
        """Not used in SAM - use first_step() and second_step() instead."""
        raise NotImplementedError("SAM requires first_step() and second_step(), not step()")
    
    def zero_grad(self):
        """Zero gradients of base optimizer."""
        self.optimizer.zero_grad()
    
    def _grad_norm(self) -> torch.Tensor:
        """Compute L2 norm of gradients across all parameters."""
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.optimizer.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def state_dict(self):
        """Return state dict of base optimizer."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict into base optimizer."""
        self.optimizer.load_state_dict(state_dict)


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    Prevents overfitting by mixing true labels with uniform distribution.
    Improves model calibration and robustness.
    
    Args:
        smoothing: Smoothing factor in [0, 1] (default: 0.1)
                  0 = standard cross-entropy, 1 = uniform distribution
        num_classes: Number of classes (default: 10)
    
    Example:
        >>> criterion = LabelSmoothingLoss(smoothing=0.1, num_classes=10)
        >>> loss = criterion(logits, labels)
    """
    
    def __init__(self, smoothing: float = 0.1, num_classes: int = 10):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
        
        Returns:
            Scalar loss value
        """
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=1)
        
        # Create smoothed labels: confidence on true class, smoothing distributed to others
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), self.confidence)
        
        # KL divergence between smoothed labels and predictions
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=1))
        
        return loss


class InputGradientPenalty:
    """
    Input Gradient Penalty regularizer.
    
    Penalizes large gradients with respect to inputs, encouraging smoothness
    and improving robustness to input perturbations.
    
    Args:
        scale: Penalty scale factor (default: 0.1)
    
    Example:
        >>> regularizer = InputGradientPenalty(scale=0.1)
        >>> reg_loss = regularizer(model, inputs, labels, loss_fn)
    """
    
    def __init__(self, scale: float = 0.1):
        self.scale = scale
    
    def __call__(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable
    ) -> torch.Tensor:
        """
        Compute input gradient penalty.
        
        Args:
            model: Neural network model
            inputs: Input batch [batch_size, ...]
            labels: Labels [batch_size]
            loss_fn: Loss function
        
        Returns:
            Regularization loss (scalar)
        """
        # Enable gradient computation for inputs
        inputs_with_grad = inputs.detach().requires_grad_(True)
        
        # Forward pass
        outputs = model(inputs_with_grad)
        loss = loss_fn(outputs, labels).mean()
        
        # Compute gradient w.r.t. inputs
        input_grads = torch.autograd.grad(
            outputs=loss,
            inputs=inputs_with_grad,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # L2 norm of input gradients
        grad_norm = torch.norm(input_grads.view(inputs.size(0), -1), p=2, dim=1)
        
        # Penalty is mean squared gradient norm
        penalty = self.scale * torch.mean(grad_norm ** 2)
        
        return penalty


def get_optimizer_with_config(
    model: nn.Module,
    config: dict
) -> tuple:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Neural network model
        config: Configuration dict with keys:
                - 'optimizer': 'adam' or 'sgd'
                - 'lr': Learning rate
                - 'weight_decay': Weight decay (L2 regularization)
                - 'use_sam': Whether to wrap with SAM
                - 'sam_rho': SAM neighborhood size (if use_sam=True)
    
    Returns:
        Tuple of (optimizer, is_sam) where is_sam indicates if SAM wrapper is used
    """
    optimizer_type = config.get('optimizer', 'adam').lower()
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    use_sam = config.get('use_sam', False)
    sam_rho = config.get('sam_rho', 0.05)
    
    # Create base optimizer
    if optimizer_type == 'adam':
        base_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        base_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Wrap with SAM if requested
    if use_sam:
        optimizer = SAMOptimizer(base_optimizer, rho=sam_rho)
        return optimizer, True
    else:
        return base_optimizer, False


def get_data_augmentation_transforms(dataset_name: str):
    """
    Get data augmentation transforms for a dataset.
    
    Args:
        dataset_name: 'mnist' or 'cifar10'
    
    Returns:
        torchvision.transforms.Compose object for training data
    """
    if dataset_name.lower() == 'mnist':
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    elif dataset_name.lower() == 'cifar10':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_standard_transforms(dataset_name: str):
    """
    Get standard (non-augmented) transforms for a dataset.
    
    Args:
        dataset_name: 'mnist' or 'cifar10'
    
    Returns:
        torchvision.transforms.Compose object for test data
    """
    if dataset_name.lower() == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    elif dataset_name.lower() == 'cifar10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
