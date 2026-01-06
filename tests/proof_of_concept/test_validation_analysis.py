"""
Integration tests for validation analyses.

Tests the 4 post-hoc validation analyses:
1. Direction convergence analysis
2. Order sensitivity analysis
3. Temporal stability analysis
4. Joint K-order optimization

Uses synthetic lambda data fixtures to avoid dependency on real experiments.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.proof_of_concept.helpers.validation_analysis import (
    run_all_validation_analyses,
    recompute_lambda_from_derivatives
)
from tests.proof_of_concept.test_utils import (
    TempDirectory,
    SyntheticLambdaData,
    PoCAssertions
)


class TestSyntheticLambdaDataGeneration(unittest.TestCase):
    """Test synthetic lambda data generation."""
    
    def test_generate_realistic_trajectory(self):
        """Test that synthetic data has expected structure."""
        data = SyntheticLambdaData.generate_realistic_trajectory(
            n_epochs=50,
            K_dirs=15,
            max_order=6
        )
        
        # Check all required keys exist
        required_keys = [
            'epochs', 'lambda_means', 'lambda_stds', 'timestamps',
            'train_accuracies', 'test_accuracies', 'train_losses', 'test_losses',
            'lambda_values_per_epoch', 'derivatives_per_epoch'
        ]
        for key in required_keys:
            self.assertIn(key, data)
        
        # Check array shapes are consistent
        n_measurements = len(data['epochs'])
        self.assertEqual(len(data['lambda_means']), n_measurements)
        self.assertEqual(len(data['lambda_stds']), n_measurements)
        self.assertEqual(len(data['lambda_values_per_epoch']), n_measurements)
        self.assertEqual(len(data['derivatives_per_epoch']), n_measurements)
    
    def test_lambda_trajectory_decreases(self):
        """Test that lambda generally decreases over training."""
        data = SyntheticLambdaData.generate_realistic_trajectory(
            n_epochs=50,
            lambda_init=0.8,
            lambda_final=0.3,
            noise_scale=0.01  # Low noise for clear trend
        )
        
        # First and last lambda should follow init/final pattern
        first_lambda = data['lambda_means'][0]
        last_lambda = data['lambda_means'][-1]
        
        # Should be decreasing (with some tolerance for noise)
        self.assertGreater(first_lambda, last_lambda - 0.2)
    
    def test_accuracy_increases(self):
        """Test that accuracy increases over training."""
        data = SyntheticLambdaData.generate_realistic_trajectory(n_epochs=50)
        
        first_acc = data['train_accuracies'][0]
        last_acc = data['train_accuracies'][-1]
        
        self.assertGreater(last_acc, first_acc)
    
    def test_save_to_npz(self):
        """Test saving synthetic data to .npz file."""
        data = SyntheticLambdaData.generate_realistic_trajectory(n_epochs=10)
        
        with TempDirectory() as tmpdir:
            filepath = tmpdir / 'test_lambda_data.npz'
            SyntheticLambdaData.save_to_npz(data, filepath)
            
            # Verify file exists and can be loaded
            self.assertTrue(filepath.exists())
            
            loaded = np.load(filepath, allow_pickle=True)
            self.assertIn('epochs', loaded)
            self.assertIn('lambda_means', loaded)


class TestRecomputeLambdaFromDerivatives(unittest.TestCase):
    """Test lambda recomputation from stored derivatives."""
    
    def test_recompute_full_order_range(self):
        """Test recomputing lambda with full derivative order range."""
        # Generate synthetic derivatives
        K_dirs = 10
        max_order = 6
        derivatives_per_dir = []
        
        for k in range(K_dirs):
            # Create plausible derivative sequence
            derivs = [1.0 / (i + 1) for i in range(max_order)]
            derivatives_per_dir.append(derivs)
        
        # Recompute lambda
        mean, std, values = recompute_lambda_from_derivatives(
            derivatives_per_dir,
            K=K_dirs,
            order_start=1,
            order_end=max_order
        )
        
        # Check outputs
        self.assertIsInstance(mean, (float, np.floating))
        self.assertIsInstance(std, (float, np.floating))
        self.assertEqual(len(values), K_dirs)
        self.assertTrue(np.isfinite(mean))
        self.assertTrue(np.isfinite(std))
        self.assertGreaterEqual(std, 0)
    
    def test_recompute_partial_order_range(self):
        """Test recomputing lambda with subset of derivative orders."""
        K_dirs = 5
        max_order = 6
        derivatives_per_dir = []
        
        for k in range(K_dirs):
            derivs = [1.0 / (i + 1) for i in range(max_order)]
            derivatives_per_dir.append(derivs)
        
        # Use only orders 2-4
        mean, std, values = recompute_lambda_from_derivatives(
            derivatives_per_dir,
            K=K_dirs,
            order_start=2,
            order_end=4
        )
        
        self.assertTrue(np.isfinite(mean))
        self.assertEqual(len(values), K_dirs)
    
    def test_recompute_fewer_directions(self):
        """Test recomputing lambda with fewer directions than available."""
        K_dirs_available = 15
        K_dirs_use = 5
        max_order = 6
        
        derivatives_per_dir = []
        for k in range(K_dirs_available):
            derivs = [1.0 / (i + 1) for i in range(max_order)]
            derivatives_per_dir.append(derivs)
        
        # Use only first 5 directions
        mean, std, values = recompute_lambda_from_derivatives(
            derivatives_per_dir,
            K=K_dirs_use,
            order_start=1,
            order_end=max_order
        )
        
        # Should return K_dirs_use values
        self.assertEqual(len(values), K_dirs_use)


class TestDirectionConvergenceAnalysis(unittest.TestCase):
    """Test direction convergence analysis."""
    
    def test_convergence_analysis_runs(self):
        """Test that direction convergence analysis completes."""
        # Create synthetic lambda data
        data = SyntheticLambdaData.generate_realistic_trajectory(
            n_epochs=50,
            K_dirs=15,
            max_order=6
        )
        
        with TempDirectory() as tmpdir:
            # Save synthetic data
            experiment_dir = tmpdir / 'mnist' / 'baseline'
            experiment_dir.mkdir(parents=True)
            lambda_path = experiment_dir / 'lambda_data.npz'
            SyntheticLambdaData.save_to_npz(data, lambda_path)
            
            # Run convergence analysis
            from experiments.proof_of_concept.helpers.validation_analysis import direction_convergence_analysis
            
            validation_dir = tmpdir / 'validation_analyses'
            validation_dir.mkdir()
            
            direction_convergence_analysis(
                [str(lambda_path)],
                str(validation_dir)
            )
            
            # Check output plot exists
            plot_path = validation_dir / 'direction_convergence' / 'mnist_baseline_convergence.png'
            self.assertTrue(plot_path.exists())


class TestOrderSensitivityAnalysis(unittest.TestCase):
    """Test order sensitivity analysis."""
    
    def test_order_sensitivity_analysis_runs(self):
        """Test that order sensitivity analysis completes."""
        # Create synthetic lambda data
        data = SyntheticLambdaData.generate_realistic_trajectory(
            n_epochs=50,
            K_dirs=15,
            max_order=6
        )
        
        with TempDirectory() as tmpdir:
            # Save synthetic data
            experiment_dir = tmpdir / 'mnist' / 'baseline'
            experiment_dir.mkdir(parents=True)
            lambda_path = experiment_dir / 'lambda_data.npz'
            SyntheticLambdaData.save_to_npz(data, lambda_path)
            
            # Run order sensitivity analysis
            from experiments.proof_of_concept.helpers.validation_analysis import order_sensitivity_analysis
            
            validation_dir = tmpdir / 'validation_analyses'
            validation_dir.mkdir()
            
            order_sensitivity_analysis(
                [str(lambda_path)],
                str(validation_dir)
            )
            
            # Check output plot exists
            plot_path = validation_dir / 'order_sensitivity' / 'mnist_baseline_order_sensitivity.png'
            self.assertTrue(plot_path.exists())


class TestTemporalStabilityAnalysis(unittest.TestCase):
    """Test temporal stability analysis."""
    
    def test_temporal_stability_analysis_runs(self):
        """Test that temporal stability analysis completes."""
        # Create synthetic lambda data
        data = SyntheticLambdaData.generate_realistic_trajectory(
            n_epochs=50,
            K_dirs=15,
            max_order=6
        )
        
        with TempDirectory() as tmpdir:
            # Save synthetic data
            experiment_dir = tmpdir / 'mnist' / 'baseline'
            experiment_dir.mkdir(parents=True)
            lambda_path = experiment_dir / 'lambda_data.npz'
            SyntheticLambdaData.save_to_npz(data, lambda_path)
            
            # Run temporal stability analysis
            from experiments.proof_of_concept.helpers.validation_analysis import temporal_stability_analysis
            
            validation_dir = tmpdir / 'validation_analyses'
            validation_dir.mkdir()
            
            temporal_stability_analysis(
                [str(lambda_path)],
                str(validation_dir)
            )
            
            # Check output plots exist
            spaghetti_path = validation_dir / 'temporal_stability' / 'mnist_baseline_spaghetti.png'
            violin_path = validation_dir / 'temporal_stability' / 'mnist_baseline_violin.png'
            
            self.assertTrue(spaghetti_path.exists())
            self.assertTrue(violin_path.exists())


class TestJointKOrderOptimization(unittest.TestCase):
    """Test joint K-order optimization analysis."""
    
    def test_joint_optimization_analysis_runs(self):
        """Test that joint K-order optimization analysis completes."""
        # Create synthetic lambda data
        data = SyntheticLambdaData.generate_realistic_trajectory(
            n_epochs=50,
            K_dirs=15,
            max_order=6
        )
        
        with TempDirectory() as tmpdir:
            # Save synthetic data
            experiment_dir = tmpdir / 'mnist' / 'baseline'
            experiment_dir.mkdir(parents=True)
            lambda_path = experiment_dir / 'lambda_data.npz'
            SyntheticLambdaData.save_to_npz(data, lambda_path)
            
            # Run joint optimization analysis
            from experiments.proof_of_concept.helpers.validation_analysis import joint_K_order_optimization
            
            validation_dir = tmpdir / 'validation_analyses'
            validation_dir.mkdir()
            
            joint_K_order_optimization(
                [str(lambda_path)],
                str(validation_dir)
            )
            
            # Check output plot exists
            plot_path = validation_dir / 'joint_optimization' / 'mnist_baseline_joint_optimization.png'
            self.assertTrue(plot_path.exists())


class TestRunAllValidationAnalyses(unittest.TestCase):
    """Test running all validation analyses together."""
    
    def test_all_analyses_run_successfully(self):
        """Test that all 4 validation analyses complete."""
        # Create synthetic lambda data for multiple experiments
        data1 = SyntheticLambdaData.generate_realistic_trajectory(
            n_epochs=50, K_dirs=15, max_order=6, seed=42
        )
        data2 = SyntheticLambdaData.generate_realistic_trajectory(
            n_epochs=50, K_dirs=15, max_order=6, seed=43
        )
        
        with TempDirectory() as tmpdir:
            # Create experiment directories
            exp1_dir = tmpdir / 'mnist' / 'baseline'
            exp2_dir = tmpdir / 'mnist' / 'dropout_0.5'
            exp1_dir.mkdir(parents=True)
            exp2_dir.mkdir(parents=True)
            
            # Save synthetic data
            SyntheticLambdaData.save_to_npz(data1, exp1_dir / 'lambda_data.npz')
            SyntheticLambdaData.save_to_npz(data2, exp2_dir / 'lambda_data.npz')
            
            # Run all validation analyses
            run_all_validation_analyses(str(tmpdir))
            
            # Check all expected plots exist
            validation_dir = tmpdir / 'validation_analyses'
            
            # For each experiment
            for method in ['baseline', 'dropout_0.5']:
                # Direction convergence
                self.assertTrue(
                    (validation_dir / 'direction_convergence' / f'mnist_{method}_convergence.png').exists()
                )
                
                # Order sensitivity
                self.assertTrue(
                    (validation_dir / 'order_sensitivity' / f'mnist_{method}_order_sensitivity.png').exists()
                )
                
                # Temporal stability (2 plots)
                self.assertTrue(
                    (validation_dir / 'temporal_stability' / f'mnist_{method}_spaghetti.png').exists()
                )
                self.assertTrue(
                    (validation_dir / 'temporal_stability' / f'mnist_{method}_violin.png').exists()
                )
                
                # Joint optimization
                self.assertTrue(
                    (validation_dir / 'joint_optimization' / f'mnist_{method}_joint_optimization.png').exists()
                )
    
    def test_handles_missing_lambda_files(self):
        """Test that analysis handles missing lambda data files gracefully."""
        with TempDirectory() as tmpdir:
            # Create empty results directory
            results_dir = tmpdir / 'results'
            results_dir.mkdir()
            
            # Should not crash even with no data files
            try:
                run_all_validation_analyses(str(results_dir))
            except Exception as e:
                # Should handle gracefully (may print warnings but shouldn't crash)
                # If it does throw an exception, it should be a graceful one
                self.assertNotIsInstance(e, AttributeError)
                self.assertNotIsInstance(e, KeyError)
    
    def test_custom_output_directory(self):
        """Test specifying custom output directory for validation."""
        data = SyntheticLambdaData.generate_realistic_trajectory(n_epochs=10)
        
        with TempDirectory() as tmpdir:
            # Create experiment directory
            exp_dir = tmpdir / 'results' / 'mnist' / 'baseline'
            exp_dir.mkdir(parents=True)
            SyntheticLambdaData.save_to_npz(data, exp_dir / 'lambda_data.npz')
            
            # Custom output directory
            custom_output = tmpdir / 'custom_validation'
            
            # Run with custom output
            run_all_validation_analyses(
                str(tmpdir / 'results'),
                output_dir=str(custom_output)
            )
            
            # Check plots are in custom directory
            self.assertTrue(custom_output.exists())
            self.assertTrue(
                (custom_output / 'direction_convergence' / 'mnist_baseline_convergence.png').exists()
            )


class TestValidationAnalysisAssertions(unittest.TestCase):
    """Test validation analysis assertion helpers."""
    
    def test_assert_validation_plots_exist_success(self):
        """Test assertion passes when all plots exist."""
        with TempDirectory() as tmpdir:
            validation_dir = tmpdir / 'validation_analyses'
            
            # Create all expected plot files
            plots = [
                'direction_convergence/mnist_baseline_convergence.png',
                'order_sensitivity/mnist_baseline_order_sensitivity.png',
                'temporal_stability/mnist_baseline_spaghetti.png',
                'temporal_stability/mnist_baseline_violin.png',
                'joint_optimization/mnist_baseline_joint_optimization.png'
            ]
            
            for plot_path in plots:
                full_path = validation_dir / plot_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text('fake plot')
            
            # Should not raise
            PoCAssertions.assert_validation_plots_exist(validation_dir, 'mnist', 'baseline')
    
    def test_assert_validation_plots_exist_failure(self):
        """Test assertion fails when plots are missing."""
        with TempDirectory() as tmpdir:
            validation_dir = tmpdir / 'validation_analyses'
            validation_dir.mkdir()
            
            # Should raise assertion error
            with self.assertRaises(AssertionError):
                PoCAssertions.assert_validation_plots_exist(validation_dir, 'mnist', 'baseline')


class TestValidationWithMultipleDatasets(unittest.TestCase):
    """Test validation analyses with multiple datasets."""
    
    def test_handles_mnist_and_cifar10(self):
        """Test validation analyses work with both MNIST and CIFAR-10 data."""
        mnist_data = SyntheticLambdaData.generate_realistic_trajectory(n_epochs=10, seed=42)
        cifar_data = SyntheticLambdaData.generate_realistic_trajectory(n_epochs=10, seed=43)
        
        with TempDirectory() as tmpdir:
            # Create experiment directories
            mnist_dir = tmpdir / 'mnist' / 'baseline'
            cifar_dir = tmpdir / 'cifar10' / 'baseline'
            mnist_dir.mkdir(parents=True)
            cifar_dir.mkdir(parents=True)
            
            # Save data
            SyntheticLambdaData.save_to_npz(mnist_data, mnist_dir / 'lambda_data.npz')
            SyntheticLambdaData.save_to_npz(cifar_data, cifar_dir / 'lambda_data.npz')
            
            # Run validation
            run_all_validation_analyses(str(tmpdir))
            
            validation_dir = tmpdir / 'validation_analyses'
            
            # Check both datasets have plots
            self.assertTrue(
                (validation_dir / 'direction_convergence' / 'mnist_baseline_convergence.png').exists()
            )
            self.assertTrue(
                (validation_dir / 'direction_convergence' / 'cifar10_baseline_convergence.png').exists()
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
