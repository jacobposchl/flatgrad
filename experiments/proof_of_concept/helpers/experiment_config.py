"""
Experiment configuration system for proof-of-concept experiments.

Defines all experiment configurations across different regularization methods.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ExperimentConfig:
    """
    Configuration for a single proof-of-concept experiment.
    
    Attributes:
        method_name: Name of the regularization method (e.g., 'baseline', 'dropout_0.3', 'sam_0.05')
        dataset: Dataset name ('mnist' or 'cifar10')
        model_type: Model architecture ('mnist_convnet' or 'cifar10_convnet')
        
        # Training hyperparameters
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        optimizer: Optimizer type ('adam', 'sgd', or 'adamw')
        
        # Regularization parameters
        regularization_type: Type of regularization ('none', 'dropout', 'weight_decay', 
                            'augmentation', 'label_smoothing', 'sam', 'igp')
        dropout_rate: Dropout rate (if using dropout)
        weight_decay: Weight decay coefficient (if using weight decay)
        use_augmentation: Whether to use data augmentation
        label_smoothing: Label smoothing factor (if using label smoothing)
        sam_rho: SAM neighborhood size (if using SAM)
        igp_scale: Input gradient penalty scale (if using IGP)
        
        # Lambda measurement parameters
        K_dirs: Number of random directions for lambda estimation
        max_order: Maximum derivative order
        
        # Data subset parameters
        train_subset_size: Number of training samples to use (None = full dataset)
        test_subset_size: Number of test samples to use (None = full dataset)
    """
    
    # Experiment identification
    method_name: str
    dataset: str
    model_type: str
    
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 128
    lr: float = 0.001
    optimizer: str = 'adam'
    use_lr_scheduler: bool = False  # Whether to use cosine annealing LR scheduler
    
    # Regularization parameters
    regularization_type: str = 'none'
    dropout_rate: float = 0.0
    weight_decay: float = 0.0
    use_augmentation: bool = False
    label_smoothing: float = 0.0
    sam_rho: float = 0.0
    igp_scale: float = 0.0
    
    # Lambda measurement parameters
    K_dirs: int = 15
    max_order: int = 6
    
    # Data subset parameters
    train_subset_size: Optional[int] = 5000
    test_subset_size: Optional[int] = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, output_path: str):
        """Save config to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, config_path: str) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_all_experiment_configs(dataset: str = 'both') -> list[ExperimentConfig]:
    """
    Generate all experiment configurations for proof-of-concept experiments.
    
    Uses dataset-specific hyperparameters to ensure both datasets operate in the 
    "Goldilocks zone" where regularization effects are observable:
    - MNIST: Reduced subset (1000 train) to prevent ceiling effect
    - CIFAR10: Higher LR, more epochs, and LR scheduling for better convergence
    
    Args:
        dataset: Which dataset(s) to generate configs for ('mnist', 'cifar10', or 'both')
    
    Returns:
        List of ExperimentConfig objects
    """
    datasets_to_run = []
    if dataset in ['mnist', 'both']:
        datasets_to_run.append(('mnist', 'mnist_mlp'))  # Use large MLP to enable overfitting
    if dataset in ['cifar10', 'both']:
        datasets_to_run.append(('cifar10', 'cifar10_mlp'))  # Use MLP instead of CNN
    
    configs = []
    
    for ds_name, model_type in datasets_to_run:
        # Dataset-specific hyperparameters
        if ds_name == 'mnist':
            base_lr = 0.001
            base_epochs = 100  # Increased from 50 to allow overfitting
            train_size = 5000  # Realistic dataset size
            test_size = 1000
            use_scheduler = False
        else:  # cifar10
            base_lr = 0.001  # Same as MNIST
            base_epochs = 50  # Shorter - MLP still learns too fast
            train_size = 5000  # Reduced from 10k - smaller MLP needs less data
            test_size = 1000
            use_scheduler = False  # Disable scheduler for faster experiments
        
        # 1. Baseline (no regularization)
        configs.append(ExperimentConfig(
            method_name='baseline',
            dataset=ds_name,
            model_type=model_type,
            regularization_type='none',
            lr=base_lr,
            epochs=base_epochs,
            train_subset_size=train_size,
            test_subset_size=test_size,
            use_lr_scheduler=use_scheduler
        ))
        
        # 2. Dropout (3 rates: 0.3, 0.5, 0.7)
        for dropout_rate in [0.3, 0.5, 0.7]:
            configs.append(ExperimentConfig(
                method_name=f'dropout_{dropout_rate}',
                dataset=ds_name,
                model_type=model_type,
                regularization_type='dropout',
                dropout_rate=dropout_rate,
                lr=base_lr,
                epochs=base_epochs,
                train_subset_size=train_size,
                test_subset_size=test_size,
                use_lr_scheduler=use_scheduler
            ))
        
        # 3. Weight Decay (3 scales: 0.0001, 0.001, 0.01)
        for wd in [0.0001, 0.001, 0.01]:
            configs.append(ExperimentConfig(
                method_name=f'weight_decay_{wd}',
                dataset=ds_name,
                model_type=model_type,
                regularization_type='weight_decay',
                weight_decay=wd,
                lr=base_lr,
                epochs=base_epochs,
                train_subset_size=train_size,
                test_subset_size=test_size,
                use_lr_scheduler=use_scheduler
            ))
        
        # 4. Data Augmentation
        configs.append(ExperimentConfig(
            method_name='augmentation',
            dataset=ds_name,
            model_type=model_type,
            regularization_type='augmentation',
            use_augmentation=True,
            lr=base_lr,
            epochs=base_epochs,
            train_subset_size=train_size,
            test_subset_size=test_size,
            use_lr_scheduler=use_scheduler
        ))
        
        # 5. Label Smoothing (3 values: 0.05, 0.1, 0.15)
        for smoothing in [0.05, 0.1, 0.15]:
            configs.append(ExperimentConfig(
                method_name=f'label_smoothing_{smoothing}',
                dataset=ds_name,
                model_type=model_type,
                regularization_type='label_smoothing',
                label_smoothing=smoothing,
                lr=base_lr,
                epochs=base_epochs,
                train_subset_size=train_size,
                test_subset_size=test_size,
                use_lr_scheduler=use_scheduler
            ))
        
        # 6. SAM (3 rho values: 0.05, 0.1, 0.2)
        for rho in [0.05, 0.1, 0.2]:
            configs.append(ExperimentConfig(
                method_name=f'sam_{rho}',
                dataset=ds_name,
                model_type=model_type,
                regularization_type='sam',
                sam_rho=rho,
                lr=base_lr,
                epochs=base_epochs,
                train_subset_size=train_size,
                test_subset_size=test_size,
                use_lr_scheduler=use_scheduler
            ))
        
        # 7. Input Gradient Penalty (3 scales: 0.01, 0.1, 1.0)
        for scale in [0.01, 0.1, 1.0]:
            configs.append(ExperimentConfig(
                method_name=f'igp_{scale}',
                dataset=ds_name,
                model_type=model_type,
                regularization_type='igp',
                igp_scale=scale,
                lr=base_lr,
                epochs=base_epochs,
                train_subset_size=train_size,
                test_subset_size=test_size,
                use_lr_scheduler=use_scheduler
            ))
    
    return configs


def get_config_by_name(method_name: str, dataset: str) -> Optional[ExperimentConfig]:
    """
    Get a specific experiment configuration by method name and dataset.
    
    Args:
        method_name: Name of the method (e.g., 'baseline', 'dropout_0.5', 'sam_0.05')
        dataset: Dataset name ('mnist' or 'cifar10')
    
    Returns:
        ExperimentConfig if found, None otherwise
    """
    all_configs = get_all_experiment_configs(dataset=dataset)
    
    for config in all_configs:
        if config.method_name == method_name and config.dataset == dataset:
            return config
    
    return None


def save_all_configs(output_dir: str, dataset: str = 'both'):
    """
    Save all experiment configurations to individual JSON files.
    
    Args:
        output_dir: Directory to save config files
        dataset: Which dataset(s) to generate configs for
    """
    configs = get_all_experiment_configs(dataset=dataset)
    
    for config in configs:
        filename = f"{config.dataset}_{config.method_name}.json"
        output_path = Path(output_dir) / filename
        config.save(str(output_path))
    
    print(f"Saved {len(configs)} experiment configurations to {output_dir}")
