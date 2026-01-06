"""
Integration tests for the training pipeline.

Tests the complete training loop with all 7 regularization methods,
lambda measurement, checkpoint handling, and output file generation.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.proof_of_concept.helpers.training_pipeline import (
    run_single_experiment,
    adaptive_measurement_schedule
)
from tests.proof_of_concept.test_utils import (
    MinimalConfig,
    TempDirectory,
    MockDataLoaderFactory,
    PoCAssertions
)


class TestAdaptiveMeasurementSchedule(unittest.TestCase):
    """Test lambda measurement scheduling."""
    
    def test_schedule_10_epochs(self):
        """Test schedule for 10 epochs."""
        schedule = adaptive_measurement_schedule(10)
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(schedule, expected)
    
    def test_schedule_30_epochs(self):
        """Test schedule for 30 epochs."""
        schedule = adaptive_measurement_schedule(30)
        # Epochs 0-10: every epoch
        # Epochs 11-30: every 2 epochs
        expected = list(range(11)) + list(range(12, 31, 2))
        self.assertEqual(schedule, expected)
    
    def test_schedule_50_epochs(self):
        """Test schedule for 50 epochs (full)."""
        schedule = adaptive_measurement_schedule(50)
        # Epochs 0-10: every epoch
        # Epochs 11-30: every 2 epochs
        # Epochs 31-50: every 5 epochs
        expected = (
            list(range(11)) +           # 0-10
            list(range(12, 31, 2)) +    # 12, 14, ..., 30
            list(range(35, 51, 5))      # 35, 40, 45, 50
        )
        self.assertEqual(schedule, expected)
    
    def test_schedule_2_epochs(self):
        """Test schedule for minimal epochs."""
        schedule = adaptive_measurement_schedule(2)
        expected = [0, 1, 2]
        self.assertEqual(schedule, expected)


class TestBaselineTraining(unittest.TestCase):
    """Test baseline (no regularization) training."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_baseline_mnist_completes(self, mock_data_loaders):
        """Test that baseline MNIST training completes successfully."""
        config = MinimalConfig.baseline_mnist(epochs=2)
        
        # Mock data loaders
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        # Verify result structure
        PoCAssertions.assert_experiment_result_valid(result)
        
        # Verify config matches
        self.assertEqual(result['config'].method_name, 'baseline')
        self.assertEqual(result['config'].dataset, 'mnist')
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_baseline_cifar10_completes(self, mock_data_loaders):
        """Test that baseline CIFAR-10 training completes successfully."""
        config = MinimalConfig.baseline_cifar10(epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        PoCAssertions.assert_experiment_result_valid(result)
        self.assertEqual(result['config'].dataset, 'cifar10')
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_output_files_created(self, mock_data_loaders):
        """Test that all expected output files are created."""
        config = MinimalConfig.baseline_mnist(epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
            output_dir = Path(result['output_dir'])
            
            # Verify files exist
            PoCAssertions.assert_output_files_exist(output_dir, expect_checkpoint=False)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_lambda_data_structure(self, mock_data_loaders):
        """Test lambda_data.npz contains expected data."""
        config = MinimalConfig.baseline_mnist(epochs=2, K_dirs=3, max_order=4)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
            lambda_path = Path(result['lambda_data_path'])
            
            # Verify lambda data structure
            PoCAssertions.assert_lambda_npz_valid(lambda_path)
            
            # Additional checks
            data = np.load(lambda_path, allow_pickle=True)
            # Should have measurements at epochs [0, 1, 2]
            self.assertTrue(len(data['epochs']) >= 2)  # At least start and end


class TestDropoutRegularization(unittest.TestCase):
    """Test dropout regularization."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_dropout_0_3(self, mock_data_loaders):
        """Test dropout rate 0.3."""
        config = MinimalConfig.dropout_mnist(dropout_rate=0.3, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        PoCAssertions.assert_experiment_result_valid(result)
        self.assertIn('dropout', result['config'].method_name)
        self.assertEqual(result['config'].dropout_rate, 0.3)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_dropout_0_5(self, mock_data_loaders):
        """Test dropout rate 0.5."""
        config = MinimalConfig.dropout_mnist(dropout_rate=0.5, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        self.assertEqual(result['config'].dropout_rate, 0.5)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_dropout_0_7(self, mock_data_loaders):
        """Test dropout rate 0.7."""
        config = MinimalConfig.dropout_mnist(dropout_rate=0.7, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        self.assertEqual(result['config'].dropout_rate, 0.7)


class TestWeightDecayRegularization(unittest.TestCase):
    """Test weight decay (L2) regularization."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_weight_decay_0_0001(self, mock_data_loaders):
        """Test weight decay 0.0001."""
        config = MinimalConfig.weight_decay_mnist(weight_decay=0.0001, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        PoCAssertions.assert_experiment_result_valid(result)
        self.assertEqual(result['config'].weight_decay, 0.0001)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_weight_decay_0_001(self, mock_data_loaders):
        """Test weight decay 0.001."""
        config = MinimalConfig.weight_decay_mnist(weight_decay=0.001, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        self.assertEqual(result['config'].weight_decay, 0.001)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_weight_decay_0_01(self, mock_data_loaders):
        """Test weight decay 0.01."""
        config = MinimalConfig.weight_decay_mnist(weight_decay=0.01, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        self.assertEqual(result['config'].weight_decay, 0.01)


class TestAugmentationRegularization(unittest.TestCase):
    """Test data augmentation regularization."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_augmentation_mnist(self, mock_data_loaders):
        """Test data augmentation for MNIST."""
        config = MinimalConfig.augmentation_mnist(epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        PoCAssertions.assert_experiment_result_valid(result)
        self.assertTrue(result['config'].use_augmentation)


class TestLabelSmoothingRegularization(unittest.TestCase):
    """Test label smoothing regularization."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_label_smoothing_0_05(self, mock_data_loaders):
        """Test label smoothing 0.05."""
        config = MinimalConfig.label_smoothing_mnist(smoothing=0.05, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        PoCAssertions.assert_experiment_result_valid(result)
        self.assertEqual(result['config'].label_smoothing, 0.05)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_label_smoothing_0_1(self, mock_data_loaders):
        """Test label smoothing 0.1."""
        config = MinimalConfig.label_smoothing_mnist(smoothing=0.1, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        self.assertEqual(result['config'].label_smoothing, 0.1)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_label_smoothing_0_15(self, mock_data_loaders):
        """Test label smoothing 0.15."""
        config = MinimalConfig.label_smoothing_mnist(smoothing=0.15, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        self.assertEqual(result['config'].label_smoothing, 0.15)


class TestSAMRegularization(unittest.TestCase):
    """Test SAM (Sharpness-Aware Minimization) regularization."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_sam_0_05(self, mock_data_loaders):
        """Test SAM rho 0.05."""
        config = MinimalConfig.sam_mnist(rho=0.05, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        PoCAssertions.assert_experiment_result_valid(result)
        self.assertEqual(result['config'].sam_rho, 0.05)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_sam_0_1(self, mock_data_loaders):
        """Test SAM rho 0.1."""
        config = MinimalConfig.sam_mnist(rho=0.1, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        self.assertEqual(result['config'].sam_rho, 0.1)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_sam_0_2(self, mock_data_loaders):
        """Test SAM rho 0.2."""
        config = MinimalConfig.sam_mnist(rho=0.2, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        self.assertEqual(result['config'].sam_rho, 0.2)


class TestIGPRegularization(unittest.TestCase):
    """Test IGP (Input Gradient Penalty) regularization."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_igp_0_01(self, mock_data_loaders):
        """Test IGP scale 0.01."""
        config = MinimalConfig.igp_mnist(scale=0.01, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        PoCAssertions.assert_experiment_result_valid(result)
        self.assertEqual(result['config'].igp_scale, 0.01)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_igp_0_1(self, mock_data_loaders):
        """Test IGP scale 0.1."""
        config = MinimalConfig.igp_mnist(scale=0.1, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        self.assertEqual(result['config'].igp_scale, 0.1)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_igp_1_0(self, mock_data_loaders):
        """Test IGP scale 1.0."""
        config = MinimalConfig.igp_mnist(scale=1.0, epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
        
        self.assertEqual(result['config'].igp_scale, 1.0)


class TestLambdaMeasurement(unittest.TestCase):
    """Test lambda measurement during training."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_lambda_measured_at_correct_epochs(self, mock_data_loaders):
        """Test that lambda is measured at scheduled epochs."""
        config = MinimalConfig.baseline_mnist(epochs=10, K_dirs=3, max_order=4)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
            
            # Load lambda data
            data = np.load(result['lambda_data_path'], allow_pickle=True)
            measured_epochs = data['epochs'].tolist()
            
            # Should include epoch 0 and 10 at minimum
            self.assertIn(0, measured_epochs)
            self.assertIn(10, measured_epochs)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_lambda_values_reasonable(self, mock_data_loaders):
        """Test that lambda values are in reasonable range."""
        config = MinimalConfig.baseline_mnist(epochs=2, K_dirs=3, max_order=4)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
            
            # Check final lambda values
            final_lambda_mean = result['results']['final_lambda_mean']
            final_lambda_std = result['results']['final_lambda_std']
            
            # Should be finite
            self.assertTrue(np.isfinite(final_lambda_mean))
            self.assertTrue(np.isfinite(final_lambda_std))
            
            # Std should be non-negative
            self.assertGreaterEqual(final_lambda_std, 0)
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_derivatives_stored_correctly(self, mock_data_loaders):
        """Test that derivatives are stored with correct dimensions."""
        config = MinimalConfig.baseline_mnist(epochs=2, K_dirs=5, max_order=6)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
            
            # Load lambda data
            data = np.load(result['lambda_data_path'], allow_pickle=True)
            derivatives_per_epoch = data['derivatives_per_epoch']
            
            # Check structure: should be list of (K_dirs x max_order) arrays
            self.assertGreater(len(derivatives_per_epoch), 0)
            
            # Check first measurement
            first_measurement = derivatives_per_epoch[0]
            if first_measurement is not None and len(first_measurement) > 0:
                # Should have up to K_dirs directions
                self.assertLessEqual(len(first_measurement), 5)
                
                # Each direction should have max_order derivatives
                if len(first_measurement) > 0:
                    self.assertLessEqual(len(first_measurement[0]), 6)


class TestMetricsCSV(unittest.TestCase):
    """Test training metrics CSV output."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_csv_structure(self, mock_data_loaders):
        """Test that metrics CSV has correct structure."""
        config = MinimalConfig.baseline_mnist(epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
            
            # Load CSV
            df = pd.read_csv(result['metrics_csv_path'])
            
            # Check columns
            expected_columns = [
                'epoch', 'train_accuracy', 'test_accuracy',
                'train_loss', 'test_loss', 'generalization_gap',
                'ece', 'lambda_mean', 'lambda_std'
            ]
            for col in expected_columns:
                self.assertIn(col, df.columns)
            
            # Check we have data for all epochs
            self.assertEqual(len(df), config.epochs + 1)  # epochs 0 to N


class TestCheckpointHandling(unittest.TestCase):
    """Test checkpoint saving and resumption."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_checkpoint_cleanup_on_success(self, mock_data_loaders):
        """Test that checkpoint is deleted after successful completion."""
        config = MinimalConfig.baseline_mnist(epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
            
            checkpoint_path = Path(result['output_dir']) / 'checkpoint.pt'
            
            # Checkpoint should be deleted on successful completion
            self.assertFalse(checkpoint_path.exists())


class TestSummaryJSON(unittest.TestCase):
    """Test summary.json output."""
    
    @patch('experiments.proof_of_concept.helpers.training_pipeline.setup_data_loaders')
    def test_summary_json_structure(self, mock_data_loaders):
        """Test that summary.json has correct structure."""
        config = MinimalConfig.baseline_mnist(epochs=2)
        
        mock_factory = MockDataLoaderFactory()
        mock_data_loaders.side_effect = mock_factory.patch_setup_data_loaders(config)
        
        with TempDirectory() as tmpdir:
            result = run_single_experiment(config, str(tmpdir), device=torch.device('cpu'))
            
            summary_path = Path(result['output_dir']) / 'summary.json'
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            # Check required fields
            required_fields = [
                'method_name', 'dataset',
                'final_train_accuracy', 'final_test_accuracy',
                'final_generalization_gap', 'final_ece',
                'final_lambda_mean', 'final_lambda_std'
            ]
            for field in required_fields:
                self.assertIn(field, summary)


if __name__ == '__main__':
    unittest.main(verbosity=2)
