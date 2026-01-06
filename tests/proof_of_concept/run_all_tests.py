"""
Master test runner for all proof-of-concept tests.

Runs all test suites and provides a summary of results.
"""

import sys
import unittest
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_all_tests(verbosity=2):
    """
    Run all proof-of-concept tests.
    
    Args:
        verbosity: Unittest verbosity level (0=quiet, 1=normal, 2=verbose)
        
    Returns:
        TestResult object
    """
    print("="*80)
    print("PROOF OF CONCEPT TEST SUITE")
    print("="*80)
    print()
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover all tests in this directory
    suite = loader.discover(
        start_dir=str(Path(__file__).parent),
        pattern='test_*.py',
        top_level_dir=str(Path(__file__).parent.parent.parent)
    )
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time
    
    # Print summary
    print()
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Time: {elapsed:.2f} seconds")
    print("="*80)
    
    return result


def run_specific_suite(suite_name, verbosity=2):
    """
    Run a specific test suite.
    
    Args:
        suite_name: Name of test module (e.g., 'test_utils', 'test_training_pipeline')
        verbosity: Unittest verbosity level
        
    Returns:
        TestResult object
    """
    loader = unittest.TestLoader()
    
    # Import the specific test module
    module_path = f'tests.proof_of_concept.{suite_name}'
    suite = loader.loadTestsFromName(module_path)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run proof-of-concept tests')
    parser.add_argument(
        '--suite',
        type=str,
        choices=[
            'test_utils',
            'test_proof_of_concept_main',
            'test_training_pipeline',
            'test_validation_analysis',
            'test_end_to_end'
        ],
        help='Run specific test suite (default: run all)'
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Test output verbosity (0=quiet, 1=normal, 2=verbose)'
    )
    
    args = parser.parse_args()
    
    # Run tests
    if args.suite:
        print(f"\nRunning {args.suite}...\n")
        result = run_specific_suite(args.suite, verbosity=args.verbosity)
    else:
        result = run_all_tests(verbosity=args.verbosity)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
