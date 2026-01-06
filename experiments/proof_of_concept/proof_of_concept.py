"""
Proof of Concept: Lambda as a Diagnostic Metric for Model Training

This experiment investigates whether λ (curvature rate) serves as a reliable 
and interpretable diagnostic metric for neural network training quality.

Rather than manipulating λ directly, we observe its behavior across different 
well-established regularization and robustness techniques to determine if λ 
consistently reflects known desirable model properties.

Main runner script for executing experiments across:
- 7 regularization methods (baseline, dropout, weight_decay, augmentation, 
  label_smoothing, SAM, input_gradient_penalty)
- 2 datasets (MNIST, CIFAR-10)
- Rich lambda data collection (K=15 directions, order=6)
- Post-hoc validation analyses
"""

import argparse
import torch
import numpy as np
import random
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.proof_of_concept.helpers.experiment_config import get_all_experiment_configs
from experiments.proof_of_concept.helpers.training_pipeline import run_single_experiment
from experiments.proof_of_concept.helpers.validation_analysis import run_all_validation_analyses


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def run_all_experiments(dataset: str, output_base_dir: str):
    """
    Run all proof-of-concept experiments for specified dataset(s).
    
    Args:
        dataset: 'mnist', 'cifar10', or 'both'
        output_base_dir: Base directory for results
    """
    # Get all experiment configurations
    configs = get_all_experiment_configs(dataset=dataset)
    
    print("\n" + "="*80)
    print(f"PROOF OF CONCEPT: Lambda as a Diagnostic Metric")
    print("="*80)
    print(f"\nRunning {len(configs)} experiments on {dataset.upper()}")
    print(f"Output directory: {output_base_dir}")
    print(f"Device: {DEVICE}")
    print("="*80 + "\n")
    
    # Run each experiment
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Starting experiment: {config.dataset}/{config.method_name}")
        
        try:
            result = run_single_experiment(config, output_base_dir, device=DEVICE)
            results.append(result)
        except Exception as e:
            print(f"\nERROR in experiment {config.dataset}/{config.method_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print(f"Completed {len(results)}/{len(configs)} experiments successfully")
    print("="*80 + "\n")
    
    return results


def run_validation_analyses_wrapper(results_dir: str):
    """
    Run all post-hoc validation analyses on experiment results.
    
    Args:
        results_dir: Directory containing experiment results
    """
    print("\n" + "="*80)
    print("Running Post-Hoc Validation Analyses")
    print("="*80 + "\n")
    
    try:
        run_all_validation_analyses(results_dir)
    except Exception as e:
        print(f"\nERROR during validation analysis: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for proof-of-concept experiments."""
    parser = argparse.ArgumentParser(
        description='Proof of Concept: Lambda as a Diagnostic Metric for Model Training'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mnist', 'cifar10', 'both'],
        default='both',
        help='Which dataset to run experiments on (default: both)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/proof_of_concept',
        help='Base directory for experiment outputs (default: results/proof_of_concept)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip post-hoc validation analyses'
    )
    
    parser.add_argument(
        '--validation-only',
        action='store_true',
        help='Only run validation analyses on existing results (skip experiments)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=SEED,
        help=f'Random seed for reproducibility (default: {SEED})'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Run experiments (unless validation-only mode)
    if not args.validation_only:
        run_all_experiments(args.dataset, args.output_dir)
    
    # Run validation analyses (unless skipped)
    if not args.skip_validation:
        run_validation_analyses_wrapper(args.output_dir)
    
    print("\n" + "="*80)
    print("All tasks completed!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
