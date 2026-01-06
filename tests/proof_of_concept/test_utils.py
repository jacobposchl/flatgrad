"""
Test utilities and synthetic data fixtures for proof-of-concept tests.

Provides:
- MockDatasets: Lightweight synthetic MNIST/CIFAR-10 data
- MinimalConfig: Factory for reduced-scale test configurations
- Fixtures: Temporary directories and cleanup utilities
- Assertions: Custom validation helpers
"""

import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from dataclasses import replace
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.proof_of_concept.helpers.experiment_config import ExperimentConfig


# ============================================================================
# Synthetic Dataset Generators
# ============================================================================

class MockMNIST:
    """Lightweight mock MNIST dataset for testing."""
    
    @staticmethod
    def create_dataset(n_samples: int = 50, seed: int = 42) -> TensorDataset:
        """
        Create synthetic MNIST-like data.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            TensorDataset with (images, labels)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate random images [n_samples, 1, 28, 28]
        images = torch.randn(n_samples, 1, 28, 28)
        
        # Generate balanced class labels
        labels = torch.tensor([i % 10 for i in range(n_samples)], dtype=torch.long)
        
        return TensorDataset(images, labels)
    
    @staticmethod
    def create_loaders(
        train_size: int = 40,
        test_size: int = 20,
        batch_size: int = 4,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and test data loaders."""
        train_dataset = MockMNIST.create_dataset(train_size, seed)
        test_dataset = MockMNIST.create_dataset(test_size, seed + 1)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader


class MockCIFAR10:
    """Lightweight mock CIFAR-10 dataset for testing."""
    
    @staticmethod
    def create_dataset(n_samples: int = 50, seed: int = 42) -> TensorDataset:
        """
        Create synthetic CIFAR-10-like data.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            TensorDataset with (images, labels)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate random images [n_samples, 3, 32, 32]
        images = torch.randn(n_samples, 3, 32, 32)
        
        # Generate balanced class labels
        labels = torch.tensor([i % 10 for i in range(n_samples)], dtype=torch.long)
        
        return TensorDataset(images, labels)
    
    @staticmethod
    def create_loaders(
        train_size: int = 40,
        test_size: int = 20,
        batch_size: int = 4,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and test data loaders."""
        train_dataset = MockCIFAR10.create_dataset(train_size, seed)
        test_dataset = MockCIFAR10.create_dataset(test_size, seed + 1)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader


# ============================================================================
# Minimal Test Configurations
# ============================================================================

class MinimalConfig:
    """Factory for creating minimal test configurations."""
    
    @staticmethod
    def baseline_mnist(
        epochs: int = 2,
        batch_size: int = 4,
        train_size: int = 20,
        test_size: int = 10,
        K_dirs: int = 3,
        max_order: int = 4
    ) -> ExperimentConfig:
        """Create minimal baseline MNIST config for testing."""
        return ExperimentConfig(
            method_name='baseline',
            dataset='mnist',
            model_type='mnist_convnet',
            epochs=epochs,
            batch_size=batch_size,
            lr=0.001,
            optimizer='adam',
            regularization_type='none',
            K_dirs=K_dirs,
            max_order=max_order,
            train_subset_size=train_size,
            test_subset_size=test_size
        )
    
    @staticmethod
    def baseline_cifar10(
        epochs: int = 2,
        batch_size: int = 4,
        train_size: int = 20,
        test_size: int = 10,
        K_dirs: int = 3,
        max_order: int = 4
    ) -> ExperimentConfig:
        """Create minimal baseline CIFAR-10 config for testing."""
        return ExperimentConfig(
            method_name='baseline',
            dataset='cifar10',
            model_type='cifar10_convnet',
            epochs=epochs,
            batch_size=batch_size,
            lr=0.001,
            optimizer='adam',
            regularization_type='none',
            K_dirs=K_dirs,
            max_order=max_order,
            train_subset_size=train_size,
            test_subset_size=test_size
        )
    
    @staticmethod
    def dropout_mnist(
        dropout_rate: float = 0.5,
        epochs: int = 2,
        **kwargs
    ) -> ExperimentConfig:
        """Create minimal dropout MNIST config for testing."""
        base = MinimalConfig.baseline_mnist(epochs=epochs, **kwargs)
        return replace(
            base,
            method_name=f'dropout_{dropout_rate}',
            regularization_type='dropout',
            dropout_rate=dropout_rate
        )
    
    @staticmethod
    def weight_decay_mnist(
        weight_decay: float = 0.001,
        epochs: int = 2,
        **kwargs
    ) -> ExperimentConfig:
        """Create minimal weight decay MNIST config for testing."""
        base = MinimalConfig.baseline_mnist(epochs=epochs, **kwargs)
        return replace(
            base,
            method_name=f'weight_decay_{weight_decay}',
            regularization_type='weight_decay',
            weight_decay=weight_decay
        )
    
    @staticmethod
    def augmentation_mnist(
        epochs: int = 2,
        **kwargs
    ) -> ExperimentConfig:
        """Create minimal augmentation MNIST config for testing."""
        base = MinimalConfig.baseline_mnist(epochs=epochs, **kwargs)
        return replace(
            base,
            method_name='augmentation',
            regularization_type='augmentation',
            use_augmentation=True
        )
    
    @staticmethod
    def label_smoothing_mnist(
        smoothing: float = 0.1,
        epochs: int = 2,
        **kwargs
    ) -> ExperimentConfig:
        """Create minimal label smoothing MNIST config for testing."""
        base = MinimalConfig.baseline_mnist(epochs=epochs, **kwargs)
        return replace(
            base,
            method_name=f'label_smoothing_{smoothing}',
            regularization_type='label_smoothing',
            label_smoothing=smoothing
        )
    
    @staticmethod
    def sam_mnist(
        rho: float = 0.05,
        epochs: int = 2,
        **kwargs
    ) -> ExperimentConfig:
        """Create minimal SAM MNIST config for testing."""
        base = MinimalConfig.baseline_mnist(epochs=epochs, **kwargs)
        return replace(
            base,
            method_name=f'sam_{rho}',
            regularization_type='sam',
            sam_rho=rho
        )
    
    @staticmethod
    def igp_mnist(
        scale: float = 0.1,
        epochs: int = 2,
        **kwargs
    ) -> ExperimentConfig:
        """Create minimal IGP MNIST config for testing."""
        base = MinimalConfig.baseline_mnist(epochs=epochs, **kwargs)
        return replace(
            base,
            method_name=f'igp_{scale}',
            regularization_type='igp',
            igp_scale=scale
        )


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

class TempDirectory:
    """Context manager for temporary test directories."""
    
    def __init__(self, prefix: str = 'test_poc_'):
        self.prefix = prefix
        self.path = None
    
    def __enter__(self) -> Path:
        """Create temporary directory."""
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary directory."""
        if self.path and self.path.exists():
            shutil.rmtree(self.path)


# ============================================================================
# Assertion Helpers
# ============================================================================

class PoCAssertions:
    """Custom assertions for proof-of-concept tests."""
    
    @staticmethod
    def assert_lambda_data_valid(data: dict):
        """Assert lambda measurement data is valid."""
        required_keys = [
            'lambda_mean', 'lambda_std', 'lambda_values',
            'n_valid_directions'
        ]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"
        
        assert isinstance(data['lambda_mean'], (float, np.floating)), \
            f"lambda_mean should be float, got {type(data['lambda_mean'])}"
        assert isinstance(data['lambda_std'], (float, np.floating)), \
            f"lambda_std should be float, got {type(data['lambda_std'])}"
        assert isinstance(data['lambda_values'], (list, np.ndarray)), \
            f"lambda_values should be list/array, got {type(data['lambda_values'])}"
        assert data['n_valid_directions'] > 0, \
            "Should have at least one valid direction"
    
    @staticmethod
    def assert_experiment_result_valid(result: dict):
        """Assert experiment result structure is valid."""
        assert 'config' in result, "Missing config in result"
        assert 'results' in result, "Missing results in result"
        assert 'lambda_data_path' in result, "Missing lambda_data_path in result"
        assert 'metrics_csv_path' in result, "Missing metrics_csv_path in result"
        assert 'output_dir' in result, "Missing output_dir in result"
        
        # Check results dictionary
        results = result['results']
        required_metrics = [
            'method_name', 'dataset',
            'final_train_accuracy', 'final_test_accuracy',
            'final_generalization_gap', 'final_ece',
            'final_lambda_mean', 'final_lambda_std'
        ]
        for metric in required_metrics:
            assert metric in results, f"Missing metric in results: {metric}"
    
    @staticmethod
    def assert_output_files_exist(output_dir: Path, expect_checkpoint: bool = False):
        """Assert expected output files exist."""
        output_dir = Path(output_dir)
        
        expected_files = [
            'config.json',
            'lambda_data.npz',
            'training_metrics.csv',
            'summary.json'
        ]
        
        for filename in expected_files:
            filepath = output_dir / filename
            assert filepath.exists(), f"Missing expected file: {filename}"
        
        if expect_checkpoint:
            checkpoint_path = output_dir / 'checkpoint.pt'
            assert checkpoint_path.exists(), "Missing checkpoint file"
    
    @staticmethod
    def assert_lambda_npz_valid(npz_path: Path):
        """Assert lambda_data.npz file is valid."""
        assert npz_path.exists(), f"Lambda data file not found: {npz_path}"
        
        data = np.load(npz_path, allow_pickle=True)
        
        required_keys = [
            'epochs', 'lambda_means', 'lambda_stds', 'timestamps',
            'train_accuracies', 'test_accuracies', 'train_losses', 'test_losses',
            'lambda_values_per_epoch', 'derivatives_per_epoch'
        ]
        
        for key in required_keys:
            assert key in data, f"Missing key in lambda_data.npz: {key}"
        
        # Check array shapes are consistent
        n_measurements = len(data['epochs'])
        assert len(data['lambda_means']) == n_measurements
        assert len(data['lambda_stds']) == n_measurements
        assert len(data['train_accuracies']) == n_measurements
        assert len(data['test_accuracies']) == n_measurements
    
    @staticmethod
    def assert_validation_plots_exist(validation_dir: Path, dataset: str, method: str):
        """Assert validation analysis plots exist."""
        validation_dir = Path(validation_dir)
        
        # Expected plot files
        expected_plots = [
            f'direction_convergence/{dataset}_{method}_convergence.png',
            f'order_sensitivity/{dataset}_{method}_order_sensitivity.png',
            f'temporal_stability/{dataset}_{method}_spaghetti.png',
            f'temporal_stability/{dataset}_{method}_violin.png',
            f'joint_optimization/{dataset}_{method}_joint_optimization.png'
        ]
        
        for plot_path in expected_plots:
            full_path = validation_dir / plot_path
            assert full_path.exists(), f"Missing validation plot: {plot_path}"


# ============================================================================
# Mock Data Loader Factory (for patching)
# ============================================================================

class MockDataLoaderFactory:
    """Factory for creating mock data loaders to patch torchvision downloads."""
    
    @staticmethod
    def patch_setup_data_loaders(config: ExperimentConfig):
        """
        Create a patched version of setup_data_loaders that uses mock data.
        
        This can be used with unittest.mock.patch to replace the real
        data loading function.
        """
        def mock_setup_data_loaders(cfg):
            """Mock data loader setup."""
            if cfg.dataset == 'mnist':
                return MockMNIST.create_loaders(
                    train_size=cfg.train_subset_size or 40,
                    test_size=cfg.test_subset_size or 20,
                    batch_size=cfg.batch_size
                )
            elif cfg.dataset == 'cifar10':
                return MockCIFAR10.create_loaders(
                    train_size=cfg.train_subset_size or 40,
                    test_size=cfg.test_subset_size or 20,
                    batch_size=cfg.batch_size
                )
            else:
                raise ValueError(f"Unknown dataset: {cfg.dataset}")
        
        return mock_setup_data_loaders


# ============================================================================
# Synthetic Lambda Data Generator
# ============================================================================

class SyntheticLambdaData:
    """Generate synthetic lambda data for validation analysis tests."""
    
    @staticmethod
    def generate_realistic_trajectory(
        n_epochs: int = 50,
        K_dirs: int = 15,
        max_order: int = 6,
        lambda_init: float = 0.8,
        lambda_final: float = 0.3,
        noise_scale: float = 0.05,
        seed: int = 42
    ) -> dict:
        """
        Generate realistic lambda trajectory data.
        
        Simulates lambda decreasing over training (indicating flattening).
        """
        np.random.seed(seed)
        
        # Measurement schedule (same as real experiments)
        epochs = []
        for e in range(n_epochs + 1):
            if e <= 10:
                epochs.append(e)
            elif e <= 30 and e % 2 == 0:
                epochs.append(e)
            elif e > 30 and e % 5 == 0:
                epochs.append(e)
        
        epochs = np.array(epochs)
        n_measurements = len(epochs)
        
        # Generate decaying lambda trajectory
        progress = epochs / n_epochs
        lambda_means = lambda_init * (1 - progress) + lambda_final * progress
        lambda_means += np.random.randn(n_measurements) * noise_scale
        
        # Generate standard deviations (decrease over time)
        lambda_stds = 0.2 * (1 - progress * 0.5) + np.random.randn(n_measurements) * 0.01
        lambda_stds = np.abs(lambda_stds)
        
        # Generate per-direction values
        lambda_values_per_epoch = []
        derivatives_per_epoch = []
        
        for i, (mean, std) in enumerate(zip(lambda_means, lambda_stds)):
            # Per-direction lambda values
            values = np.random.randn(K_dirs) * std + mean
            lambda_values_per_epoch.append(values.tolist())
            
            # Per-direction derivatives (K_dirs x max_order)
            derivatives = []
            for k in range(K_dirs):
                # Generate plausible derivative sequence
                dir_derivs = np.random.rand(max_order) * 0.5 + 0.1
                derivatives.append(dir_derivs.tolist())
            derivatives_per_epoch.append(derivatives)
        
        # Generate realistic accuracy/loss trajectories
        train_accuracies = 0.5 + 0.45 * progress + np.random.randn(n_measurements) * 0.02
        test_accuracies = 0.5 + 0.4 * progress + np.random.randn(n_measurements) * 0.03
        train_losses = 1.5 * np.exp(-progress * 2) + np.random.randn(n_measurements) * 0.05
        test_losses = 1.8 * np.exp(-progress * 1.5) + np.random.randn(n_measurements) * 0.08
        
        # Clip to valid ranges
        train_accuracies = np.clip(train_accuracies, 0, 1)
        test_accuracies = np.clip(test_accuracies, 0, 1)
        train_losses = np.clip(train_losses, 0.01, None)
        test_losses = np.clip(test_losses, 0.01, None)
        
        # Timestamps (simulated)
        timestamps = np.linspace(0, 3600, n_measurements)  # 1 hour simulation
        
        return {
            'epochs': epochs,
            'lambda_means': lambda_means,
            'lambda_stds': lambda_stds,
            'timestamps': timestamps,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'lambda_values_per_epoch': np.array(lambda_values_per_epoch, dtype=object),
            'derivatives_per_epoch': np.array(derivatives_per_epoch, dtype=object)
        }
    
    @staticmethod
    def save_to_npz(data: dict, filepath: Path):
        """Save synthetic lambda data to .npz file."""
        np.savez(filepath, **data)


if __name__ == '__main__':
    # Quick test of utilities
    print("Testing MockMNIST...")
    train_loader, test_loader = MockMNIST.create_loaders()
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    print("\nTesting MinimalConfig...")
    config = MinimalConfig.baseline_mnist()
    print(f"  Config: {config.method_name}/{config.dataset}")
    print(f"  Epochs: {config.epochs}, K_dirs: {config.K_dirs}")
    
    print("\nTesting TempDirectory...")
    with TempDirectory() as tmpdir:
        print(f"  Created: {tmpdir}")
        assert tmpdir.exists()
    print(f"  Cleaned up: {not tmpdir.exists()}")
    
    print("\nTesting SyntheticLambdaData...")
    lambda_data = SyntheticLambdaData.generate_realistic_trajectory(n_epochs=10)
    print(f"  Measurements: {len(lambda_data['epochs'])}")
    print(f"  Lambda range: [{lambda_data['lambda_means'].min():.3f}, {lambda_data['lambda_means'].max():.3f}]")
    
    print("\n✓ All utilities working!")
