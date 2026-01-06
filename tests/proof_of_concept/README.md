# Proof of Concept Test Suite

Comprehensive testing suite for the proof-of-concept experiment pipeline that investigates λ (lambda/curvature rate) as a diagnostic metric for neural network training quality.

## Overview

This test suite ensures the entire end-to-end pipeline works flawlessly across:
- 7 regularization methods (baseline, dropout, weight_decay, augmentation, label_smoothing, SAM, IGP)
- 2 datasets (MNIST, CIFAR-10)
- Lambda measurement with K=15 directions and order=6
- 4 post-hoc validation analyses

## Test Structure

```
tests/proof_of_concept/
├── __init__.py                          # Package init
├── test_utils.py                        # Utilities and fixtures
├── test_proof_of_concept_main.py        # Main orchestration tests
├── test_training_pipeline.py            # Training pipeline integration tests
├── test_validation_analysis.py          # Validation analysis tests
├── test_end_to_end.py                   # End-to-end smoke tests
├── run_all_tests.py                     # Test runner
└── README.md                            # This file
```

## Test Coverage

### 1. **test_utils.py** - Test Utilities
- Mock datasets (MNIST/CIFAR-10 with synthetic data)
- Minimal configuration factories
- Synthetic lambda data generation
- Custom assertion helpers
- Temporary directory fixtures

### 2. **test_proof_of_concept_main.py** - Orchestration Tests (Unit)
- ✓ Command-line argument parsing
- ✓ Seed setting for reproducibility
- ✓ Experiment loop coordination
- ✓ Error handling and recovery
- ✓ Conditional execution modes (validation-only, skip-validation)
- ✓ Device selection (CPU/CUDA)

### 3. **test_training_pipeline.py** - Training Pipeline Tests (Integration)
- ✓ Adaptive lambda measurement schedule
- ✓ Baseline training (no regularization)
- ✓ Dropout regularization (rates: 0.3, 0.5, 0.7)
- ✓ Weight decay regularization (values: 0.0001, 0.001, 0.01)
- ✓ Data augmentation
- ✓ Label smoothing (values: 0.05, 0.1, 0.15)
- ✓ SAM optimizer (rho: 0.05, 0.1, 0.2)
- ✓ Input gradient penalty (scales: 0.01, 0.1, 1.0)
- ✓ Lambda measurement at correct epochs
- ✓ Output file generation (config.json, lambda_data.npz, metrics CSV, summary.json)
- ✓ Checkpoint handling

### 4. **test_validation_analysis.py** - Validation Analysis Tests (Integration)
- ✓ Synthetic lambda data generation
- ✓ Lambda recomputation from derivatives
- ✓ Direction convergence analysis
- ✓ Order sensitivity analysis
- ✓ Temporal stability analysis (spaghetti + violin plots)
- ✓ Joint K-order optimization
- ✓ Handling multiple datasets
- ✓ Custom output directories

### 5. **test_end_to_end.py** - End-to-End Tests (Smoke)
- ✓ Complete pipeline (single experiment)
- ✓ Complete pipeline (multiple experiments)
- ✓ Error recovery (continues after failures)
- ✓ Reproducibility with seed
- ✓ Memory efficiency
- ✓ Both datasets (MNIST + CIFAR-10)
- ✓ Directory structure creation
- ✓ Data validation (finite values, valid ranges)

## Running Tests

### Run All Tests
```bash
python tests/proof_of_concept/run_all_tests.py
```

### Run Specific Test Suite
```bash
# Test utilities
python tests/proof_of_concept/run_all_tests.py --suite test_utils

# Main orchestration
python tests/proof_of_concept/run_all_tests.py --suite test_proof_of_concept_main

# Training pipeline
python tests/proof_of_concept/run_all_tests.py --suite test_training_pipeline

# Validation analysis
python tests/proof_of_concept/run_all_tests.py --suite test_validation_analysis

# End-to-end
python tests/proof_of_concept/run_all_tests.py --suite test_end_to_end
```

### Run Individual Test File
```bash
python -m unittest tests.proof_of_concept.test_training_pipeline
```

### Run Specific Test Class
```bash
python -m unittest tests.proof_of_concept.test_training_pipeline.TestBaselineTraining
```

### Run Specific Test Method
```bash
python -m unittest tests.proof_of_concept.test_training_pipeline.TestBaselineTraining.test_baseline_mnist_completes
```

### Verbosity Options
```bash
# Quiet (only show summary)
python tests/proof_of_concept/run_all_tests.py --verbosity 0

# Normal (show test names)
python tests/proof_of_concept/run_all_tests.py --verbosity 1

# Verbose (show detailed output) - DEFAULT
python tests/proof_of_concept/run_all_tests.py --verbosity 2
```

## Expected Runtime

- **test_utils.py**: < 5 seconds (quick smoke tests)
- **test_proof_of_concept_main.py**: < 5 seconds (mocked, no real training)
- **test_training_pipeline.py**: 30-60 seconds (minimal epochs, synthetic data)
- **test_validation_analysis.py**: 15-30 seconds (synthetic lambda data)
- **test_end_to_end.py**: 30-90 seconds (complete pipeline)

**Total: < 3 minutes for full test suite**

## Test Philosophy

### Synthetic Data
All tests use synthetic data to:
- Avoid network downloads (no torchvision.datasets downloads)
- Run quickly (minimal samples: 20-50)
- Ensure reproducibility
- Isolate from external dependencies

### Minimal Configurations
Tests use reduced-scale configurations:
- Epochs: 2 (vs. 50 in production)
- K_dirs: 3 (vs. 15 in production)
- max_order: 4 (vs. 6 in production)
- Batch size: 4 (vs. 128 in production)
- Data size: 20 train / 10 test (vs. 5000/1000 in production)

### Mocking Strategy
- **Mock external I/O**: Data downloads, file saves (when appropriate)
- **Keep real logic**: Model training, lambda calculation, optimizer steps
- **Reason**: We want to test the actual computation, not just orchestration

### Test Types
1. **Unit tests**: Test individual functions in isolation (with mocking)
2. **Integration tests**: Test component interactions (minimal mocking)
3. **Smoke tests**: Test complete pipeline end-to-end (minimal mocking)

## Test Data Fixtures

### MockMNIST / MockCIFAR10
Lightweight synthetic datasets:
- Random Gaussian pixel values
- Balanced class labels
- Configurable size (default: 50 samples)

### MinimalConfig
Factory for creating test configurations:
- `baseline_mnist()` / `baseline_cifar10()`
- `dropout_mnist(dropout_rate=0.5)`
- `weight_decay_mnist(weight_decay=0.001)`
- `augmentation_mnist()`
- `label_smoothing_mnist(smoothing=0.1)`
- `sam_mnist(rho=0.05)`
- `igp_mnist(scale=0.1)`

### SyntheticLambdaData
Realistic lambda trajectory generator:
- Decaying lambda (flattening over training)
- Increasing accuracy
- Decreasing loss
- Per-direction values and derivatives
- Configurable epochs, K_dirs, max_order

### TempDirectory
Context manager for temporary test directories:
- Auto-cleanup after tests
- Isolated from workspace

## Custom Assertions

### PoCAssertions
Domain-specific assertions:
- `assert_lambda_data_valid(data)` - Validates lambda measurement structure
- `assert_experiment_result_valid(result)` - Validates experiment output
- `assert_output_files_exist(output_dir)` - Checks expected files created
- `assert_lambda_npz_valid(npz_path)` - Validates lambda_data.npz structure
- `assert_validation_plots_exist(dir, dataset, method)` - Checks all 5 plots

## Troubleshooting

### Test Failures

**"Missing required key: X"**
- Lambda data structure changed
- Check `save_lambda_data()` in training_pipeline.py

**"Missing expected file: X"**
- Output file not created
- Check `run_single_experiment()` save logic

**"No module named X"**
- Path issues
- Ensure `sys.path.insert(0, ...)` is correct

**Tests hang or timeout**
- Infinite loop in training
- Check measurement schedule
- Reduce epochs/K_dirs in test config

**CUDA out of memory**
- Tests should use CPU by default
- Check `device=torch.device('cpu')` in test calls

### Performance Issues

**Tests too slow**
- Reduce epochs (use 1-2)
- Reduce K_dirs (use 2-3)
- Reduce data size (use 10-20 samples)
- Use `--verbosity 0` to reduce output

**Memory leaks**
- Check model/optimizer cleanup
- Add `del model, optimizer` after experiments
- Call `torch.cuda.empty_cache()` if using GPU

## Continuous Integration

For CI/CD pipelines:

```bash
# Run with minimal output
python tests/proof_of_concept/run_all_tests.py --verbosity 1

# Check exit code
echo $?  # 0 = success, 1 = failure
```

## Adding New Tests

### For New Regularization Method

1. Add factory method to `MinimalConfig` in `test_utils.py`
2. Add test class to `test_training_pipeline.py`
3. Add to end-to-end test in `test_end_to_end.py`

### For New Validation Analysis

1. Add test class to `test_validation_analysis.py`
2. Update `assert_validation_plots_exist()` if new plots added
3. Add to `test_all_analyses_run_successfully()` in end-to-end

### For New Dataset

1. Add `MockDataset` class to `test_utils.py`
2. Add factory methods to `MinimalConfig`
3. Update `MockDataLoaderFactory.patch_setup_data_loaders()`
4. Add dataset-specific tests

## Coverage Goals

- **Line coverage**: > 80%
- **Branch coverage**: > 70%
- **Integration coverage**: All regularization methods × both datasets
- **Error paths**: Failure recovery, validation, edge cases

## Questions?

See the main [proof_of_concept.py](../../experiments/proof_of_concept/proof_of_concept.py) docstring for experiment details.

For implementation questions, check:
- [experiment_config.py](../../experiments/proof_of_concept/helpers/experiment_config.py) - Configuration
- [training_pipeline.py](../../experiments/proof_of_concept/helpers/training_pipeline.py) - Training loop
- [validation_analysis.py](../../experiments/proof_of_concept/helpers/validation_analysis.py) - Post-hoc analysis
- [regularization_methods.py](../../experiments/proof_of_concept/helpers/regularization_methods.py) - Regularizers
