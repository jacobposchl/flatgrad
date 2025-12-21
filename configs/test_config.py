"""
Configuration file for test settings.
"""
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TESTS_DIR = PROJECT_ROOT / "tests"

# Test-specific output directories
TEST_DERIVATIVES_OUTPUT_DIR = RESULTS_DIR / "tests" / "test_derivatives"
TEST_LAMBDA_ESTIMATION_OUTPUT_DIR = RESULTS_DIR / "tests" / "test_lambda_estimation"
TEST_METRICS_OUTPUT_DIR = RESULTS_DIR / "tests" / "test_metrics"
TEST_MODELS_OUTPUT_DIR = RESULTS_DIR / "tests" / "test_models"

# Create output directories
TEST_DERIVATIVES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_LAMBDA_ESTIMATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

