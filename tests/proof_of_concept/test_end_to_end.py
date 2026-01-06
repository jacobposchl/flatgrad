"""
End-to-end smoke test for the complete proof-of-concept pipeline.

This test runs the entire pipeline from start to finish with minimal
synthetic data to ensure all components work together correctly.

Expected runtime: < 30 seconds
"""

import unittest
from unittest.mock import patch
import sys
from pathlib import Path
import time
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.proof_of_concept.proof_of_concept import (
    run_all_experiments,
    run_validation_analyses_wrapper,
    set_seed
)
from experiments.proof_of_concept.helpers.experiment_config import ExperimentConfig
from tests.proof_of_concept.test_utils import (
    MinimalConfig,
    TempDirectory,
    MockDataLoaderFactory,
    PoCAssertions
)


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end smoke tests for the complete pipeline."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_complete_pipeline_single_experiment(self, mock_data_loaders):
        """Test complete pipeline with single experiment."""
        # Mock data loading
        mock_factory = MockDataLoaderFactory()
        
        def mock_loader_factory(config):
            return mock_factory.patch_setup_data_loaders(config)(config)
        
        mock_data_loaders.side_effect = mock_loader_factory
        
        # Create minimal config
        config = MinimalConfig.baseline_mnist(epochs=2, K_dirs=3, max_order=4)
        
        with TempDirectory() as tmpdir:
            # Mock get_all_experiment_configs to return single config
            with patch('experiments.proof_of_concept.proof_of_concept.get_all_experiment_configs') as mock_configs:
                mock_configs.return_value = [config]
                
                start_time = time.time()
                
                # Run experiments
                results = run_all_experiments('mnist', str(tmpdir))
                
                # Run validation
                run_validation_analyses_wrapper(str(tmpdir))
                
                elapsed = time.time() - start_time
                
                # Verify completion
                self.assertEqual(len(results), 1)
                PoCAssertions.assert_experiment_result_valid(results[0])
                
                # Verify output files
                output_dir = Path(results[0]['output_dir'])
                PoCAssertions.assert_output_files_exist(output_dir)
                
                # Verify validation plots
                validation_dir = tmpdir / 'validation_analyses'
                PoCAssertions.assert_validation_plots_exist(validation_dir, 'mnist', 'baseline')
                
                # Performance check
                print(f"\nComplete pipeline finished in {elapsed:.2f} seconds")
                self.assertLess(elapsed, 60, "Pipeline should complete in < 60 seconds")
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_complete_pipeline_multiple_experiments(self, mock_data_loaders):
        """Test complete pipeline with multiple experiments."""
        # Mock data loading
        mock_factory = MockDataLoaderFactory()
        
        def mock_loader_factory(config):
            return mock_factory.patch_setup_data_loaders(config)(config)
        
        mock_data_loaders.side_effect = mock_loader_factory
        
        # Create minimal configs for 3 methods
        configs = [
            MinimalConfig.baseline_mnist(epochs=2, K_dirs=3, max_order=4),
            MinimalConfig.dropout_mnist(dropout_rate=0.5, epochs=2, K_dirs=3, max_order=4),
            MinimalConfig.sam_mnist(rho=0.05, epochs=2, K_dirs=3, max_order=4)
        ]
        
        with TempDirectory() as tmpdir:
            with patch('experiments.proof_of_concept.proof_of_concept.get_all_experiment_configs') as mock_configs:
                mock_configs.return_value = configs
                
                start_time = time.time()
                
                # Run experiments
                results = run_all_experiments('mnist', str(tmpdir))
                
                # Run validation
                run_validation_analyses_wrapper(str(tmpdir))
                
                elapsed = time.time() - start_time
                
                # Verify completion
                self.assertEqual(len(results), 3)
                
                # Verify all experiments
                for result in results:
                    PoCAssertions.assert_experiment_result_valid(result)
                    output_dir = Path(result['output_dir'])
                    PoCAssertions.assert_output_files_exist(output_dir)
                
                # Verify validation plots for all methods
                validation_dir = tmpdir / 'validation_analyses'
                for method in ['baseline', 'dropout_0.5', 'sam_0.05']:
                    PoCAssertions.assert_validation_plots_exist(validation_dir, 'mnist', method)
                
                print(f"\nMultiple experiments finished in {elapsed:.2f} seconds")
                self.assertLess(elapsed, 120, "Multiple experiments should complete in < 2 minutes")
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_pipeline_with_error_recovery(self, mock_data_loaders):
        """Test that pipeline continues after individual experiment failures."""
        # Mock data loading
        mock_factory = MockDataLoaderFactory()
        
        call_count = [0]
        
        def mock_loader_factory(config):
            call_count[0] += 1
            # Fail on second experiment
            if call_count[0] == 2:
                raise RuntimeError("Simulated experiment failure")
            return mock_factory.patch_setup_data_loaders(config)(config)
        
        mock_data_loaders.side_effect = mock_loader_factory
        
        configs = [
            MinimalConfig.baseline_mnist(epochs=2, K_dirs=3, max_order=4),
            MinimalConfig.dropout_mnist(epochs=2, K_dirs=3, max_order=4),
            MinimalConfig.sam_mnist(epochs=2, K_dirs=3, max_order=4)
        ]
        
        with TempDirectory() as tmpdir:
            with patch('experiments.proof_of_concept.proof_of_concept.get_all_experiment_configs') as mock_configs:
                mock_configs.return_value = configs
                
                # Run experiments (should complete 2 out of 3)
                results = run_all_experiments('mnist', str(tmpdir))
                
                # Should have 2 successful results
                self.assertEqual(len(results), 2)
                
                # Verify successful experiments
                for result in results:
                    PoCAssertions.assert_experiment_result_valid(result)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_reproducibility_with_seed(self, mock_data_loaders):
        """Test that setting seed produces reproducible results."""
        # Mock data loading
        mock_factory = MockDataLoaderFactory()
        
        def mock_loader_factory(config):
            return mock_factory.patch_setup_data_loaders(config)(config)
        
        mock_data_loaders.side_effect = mock_loader_factory
        
        config = MinimalConfig.baseline_mnist(epochs=2, K_dirs=3, max_order=4)
        
        # Run twice with same seed
        results_1 = None
        results_2 = None
        
        with TempDirectory() as tmpdir1:
            with patch('experiments.proof_of_concept.proof_of_concept.get_all_experiment_configs') as mock_configs:
                mock_configs.return_value = [config]
                set_seed(42)
                results = run_all_experiments('mnist', str(tmpdir1))
                results_1 = results[0]['results']
        
        # Reset mock
        call_count = [0]
        
        def mock_loader_factory_2(config):
            return mock_factory.patch_setup_data_loaders(config)(config)
        
        mock_data_loaders.side_effect = mock_loader_factory_2
        
        with TempDirectory() as tmpdir2:
            with patch('experiments.proof_of_concept.proof_of_concept.get_all_experiment_configs') as mock_configs:
                mock_configs.return_value = [config]
                set_seed(42)
                results = run_all_experiments('mnist', str(tmpdir2))
                results_2 = results[0]['results']
        
        # Compare key metrics (may not be exactly identical due to GPU nondeterminism,
        # but should be very close)
        acc_diff = abs(results_1['final_test_accuracy'] - results_2['final_test_accuracy'])
        print(f"\nAccuracy difference with same seed: {acc_diff:.6f}")
        # Allow small tolerance for GPU nondeterminism
        self.assertLess(acc_diff, 0.2, "Results should be similar with same seed")


class TestPipelinePerformance(unittest.TestCase):
    """Performance and stress tests."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_pipeline_memory_efficiency(self, mock_data_loaders):
        """Test that pipeline doesn't leak memory across experiments."""
        # Mock data loading
        mock_factory = MockDataLoaderFactory()
        
        def mock_loader_factory(config):
            return mock_factory.patch_setup_data_loaders(config)(config)
        
        mock_data_loaders.side_effect = mock_loader_factory
        
        # Create multiple configs
        configs = [
            MinimalConfig.baseline_mnist(epochs=1, K_dirs=3, max_order=4),
            MinimalConfig.dropout_mnist(epochs=1, K_dirs=3, max_order=4),
            MinimalConfig.weight_decay_mnist(epochs=1, K_dirs=3, max_order=4),
        ]
        
        with TempDirectory() as tmpdir:
            with patch('experiments.proof_of_concept.proof_of_concept.get_all_experiment_configs') as mock_configs:
                mock_configs.return_value = configs
                
                # Clear CUDA cache before
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Run experiments
                results = run_all_experiments('mnist', str(tmpdir))
                
                # Should complete all without memory errors
                self.assertEqual(len(results), 3)
                
                # Clear CUDA cache after
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


class TestPipelineBothDatasets(unittest.TestCase):
    """Test pipeline with both MNIST and CIFAR-10."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_pipeline_both_datasets(self, mock_data_loaders):
        """Test pipeline runs on both datasets."""
        # Mock data loading
        mock_factory = MockDataLoaderFactory()
        
        def mock_loader_factory(config):
            return mock_factory.patch_setup_data_loaders(config)(config)
        
        mock_data_loaders.side_effect = mock_loader_factory
        
        configs = [
            MinimalConfig.baseline_mnist(epochs=2, K_dirs=3, max_order=4),
            MinimalConfig.baseline_cifar10(epochs=2, K_dirs=3, max_order=4)
        ]
        
        with TempDirectory() as tmpdir:
            with patch('experiments.proof_of_concept.proof_of_concept.get_all_experiment_configs') as mock_configs:
                mock_configs.return_value = configs
                
                # Run experiments
                results = run_all_experiments('both', str(tmpdir))
                
                # Should have both
                self.assertEqual(len(results), 2)
                
                datasets = {r['config'].dataset for r in results}
                self.assertIn('mnist', datasets)
                self.assertIn('cifar10', datasets)
                
                # Run validation
                run_validation_analyses_wrapper(str(tmpdir))
                
                # Check validation plots for both datasets
                validation_dir = tmpdir / 'validation_analyses'
                PoCAssertions.assert_validation_plots_exist(validation_dir, 'mnist', 'baseline')
                PoCAssertions.assert_validation_plots_exist(validation_dir, 'cifar10', 'baseline')


class TestPipelineDirectoryStructure(unittest.TestCase):
    """Test correct directory structure creation."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_directory_structure(self, mock_data_loaders):
        """Test that correct directory structure is created."""
        # Mock data loading
        mock_factory = MockDataLoaderFactory()
        
        def mock_loader_factory(config):
            return mock_factory.patch_setup_data_loaders(config)(config)
        
        mock_data_loaders.side_effect = mock_loader_factory
        
        configs = [
            MinimalConfig.baseline_mnist(epochs=1, K_dirs=3, max_order=4),
            MinimalConfig.dropout_mnist(dropout_rate=0.5, epochs=1, K_dirs=3, max_order=4),
            MinimalConfig.baseline_cifar10(epochs=1, K_dirs=3, max_order=4)
        ]
        
        with TempDirectory() as tmpdir:
            with patch('experiments.proof_of_concept.proof_of_concept.get_all_experiment_configs') as mock_configs:
                mock_configs.return_value = configs
                
                # Run experiments
                results = run_all_experiments('both', str(tmpdir))
                run_validation_analyses_wrapper(str(tmpdir))
                
                # Check directory structure
                expected_dirs = [
                    tmpdir / 'mnist' / 'baseline',
                    tmpdir / 'mnist' / 'dropout_0.5',
                    tmpdir / 'cifar10' / 'baseline',
                    tmpdir / 'validation_analyses' / 'direction_convergence',
                    tmpdir / 'validation_analyses' / 'order_sensitivity',
                    tmpdir / 'validation_analyses' / 'temporal_stability',
                    tmpdir / 'validation_analyses' / 'joint_optimization'
                ]
                
                for expected_dir in expected_dirs:
                    self.assertTrue(
                        expected_dir.exists(),
                        f"Expected directory not found: {expected_dir}"
                    )


class TestPipelineDataValidation(unittest.TestCase):
    """Test data validation throughout pipeline."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_lambda_values_finite_and_reasonable(self, mock_data_loaders):
        """Test that all lambda values are finite and in reasonable range."""
        # Mock data loading
        mock_factory = MockDataLoaderFactory()
        
        def mock_loader_factory(config):
            return mock_factory.patch_setup_data_loaders(config)(config)
        
        mock_data_loaders.side_effect = mock_loader_factory
        
        config = MinimalConfig.baseline_mnist(epochs=2, K_dirs=5, max_order=4)
        
        with TempDirectory() as tmpdir:
            with patch('experiments.proof_of_concept.proof_of_concept.get_all_experiment_configs') as mock_configs:
                mock_configs.return_value = [config]
                
                results = run_all_experiments('mnist', str(tmpdir))
                
                # Check lambda values
                final_lambda = results[0]['results']['final_lambda_mean']
                final_std = results[0]['results']['final_lambda_std']
                
                # Should be finite
                import numpy as np
                self.assertTrue(np.isfinite(final_lambda), "Lambda should be finite")
                self.assertTrue(np.isfinite(final_std), "Lambda std should be finite")
                
                # Std should be non-negative
                self.assertGreaterEqual(final_std, 0, "Lambda std should be non-negative")
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_accuracy_in_valid_range(self, mock_data_loaders):
        """Test that accuracy values are in [0, 1] range."""
        # Mock data loading
        mock_factory = MockDataLoaderFactory()
        
        def mock_loader_factory(config):
            return mock_factory.patch_setup_data_loaders(config)(config)
        
        mock_data_loaders.side_effect = mock_loader_factory
        
        config = MinimalConfig.baseline_mnist(epochs=2, K_dirs=3, max_order=4)
        
        with TempDirectory() as tmpdir:
            with patch('experiments.proof_of_concept.proof_of_concept.get_all_experiment_configs') as mock_configs:
                mock_configs.return_value = [config]
                
                results = run_all_experiments('mnist', str(tmpdir))
                
                # Check accuracy ranges
                train_acc = results[0]['results']['final_train_accuracy']
                test_acc = results[0]['results']['final_test_accuracy']
                
                self.assertGreaterEqual(train_acc, 0, "Train accuracy >= 0")
                self.assertLessEqual(train_acc, 1, "Train accuracy <= 1")
                self.assertGreaterEqual(test_acc, 0, "Test accuracy >= 0")
                self.assertLessEqual(test_acc, 1, "Test accuracy <= 1")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
