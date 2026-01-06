# Proof of Concept: Lambda as a Diagnostic Metric for Model Training

## Experiment Overview

This experiment investigates whether **λ (curvature rate)** serves as a reliable and interpretable diagnostic metric for neural network training quality. Rather than manipulating λ directly, we observe its behavior across different well-established regularization and robustness techniques to determine if λ consistently reflects known desirable model properties.

## Research Question

**Does λ provide consistent, interpretable signals about model generalization and robustness across independent training methods?**

If λ is a useful metric, it should:
- Correlate with generalization performance across different regularization strategies
- Show consistent patterns for methods that achieve similar robustness properties
- Remain stable (low variance across random directions) when models generalize well
- Provide early warning signals when models begin to overfit

## Experimental Design

### Datasets
- **MNIST**: 28×28 grayscale images (784 dimensions)
- **CIFAR-10**: 32×32 RGB images (3072 dimensions)

Both use subset training (5000 train, 1000 test) for computational efficiency.

### Regularization Methods Tested

We compare λ evolution across the following training configurations:

1. **No Regularization (Baseline)**
   - Standard training with no additional constraints
   - Establishes λ's natural trajectory

2. **Dropout Regularization**
   - Rates: 0.3, 0.5, 0.7
   - Known to improve generalization via ensemble effects

3. **Weight Decay (L2 Regularization)**
   - Scales: 0.0001, 0.001, 0.01
   - Penalizes large weights, prevents overfitting

4. **Data Augmentation**
   - Standard augmentation (rotation, crop, flip for CIFAR-10; rotation, translation for MNIST)
   - Improves robustness to input variations

5. **Label Smoothing**
   - Smoothing values: 0.05, 0.1, 0.15
   - Prevents overconfident predictions, improves calibration

6. **SAM (Sharpness-Aware Minimization)**
   - ρ values: 0.05, 0.1, 0.2
   - Explicitly seeks flat minima by perturbing weights in adversarial direction
   - State-of-the-art generalization method

7. **Input Gradient Penalty (IGP)**
   - Penalty scales: 0.01, 0.1, 1.0
   - Regularizes gradients with respect to inputs
   - Known to improve adversarial robustness

### Metrics Tracked

For each configuration, we measure:
- **Test accuracy**: Generalization performance
- **Generalization gap**: (Train accuracy - Test accuracy)
- **Expected Calibration Error (ECE)**: Prediction confidence alignment
- **Training loss**: Learning dynamics
- **λ information**: Various lambda information measurements for further analysis

### Lambda Information Measurement Protocol

At adaptive intervals during training:
1. Sample **K=15 random directions** in input space
2. Compute directional derivatives **up to order 6**
3. Store all individual directional derivatives and all order derivatives
4. Estimate λ from derivative decay rate: $d_{n+1}/d_n \approx \lambda$

**Measurement Schedule** (adaptive sampling to balance resolution and compute):
- **Epochs 0-10**: Every epoch (dense early tracking)
- **Epochs 11-30**: Every 2 epochs
- **Epochs 31-50**: Every 5 epochs
- **Epochs 50+**: Every 10 epochs

**Data Storage**: For each measurement, save:
- All 15 directions and the following for each direction:
    - All derivatives up to 6th order
    - Training epoch, train/test accuracy, train/test loss

This rich dataset enables post-hoc analysis of convergence, order sensitivity, and temporal stability **without additional training runs**.

## Hypotheses

### Hypothesis 1: Lambda Correlates with Generalization
**Prediction**: Methods that achieve better test accuracy will converge to similar λ ranges.

**Strong evidence for**: λ values cluster for well-generalizing models regardless of regularization method  
**Evidence against**: λ values are uncorrelated with test performance

### Hypothesis 2: Lambda Reflects Loss Landscape Flatness
**Prediction**: Flatter minima (better generalization) correspond to more negative λ values.

**Strong evidence for**: Dropout, weight decay, and data augmentation all push λ toward more negative values  
**Evidence against**: Different "flat" training methods yield wildly different λ values

### Hypothesis 3: Lambda Stability Indicates Training Quality
**Prediction**: Low λ variance (across random directions) indicates stable, well-behaved loss landscapes.

**Strong evidence for**: Overfitted models show high λ variance; well-regularized models show low variance  
**Evidence against**: λ variance is unrelated to generalization gap or calibration

### Hypothesis 4: Lambda Provides Early Warning Signals
**Prediction**: λ drifts or becomes unstable before test accuracy degrades (overfitting).

**Strong evidence for**: λ metrics precede accuracy degradation by several epochs  
**Evidence against**: λ changes only after overfitting is already visible in test accuracy

## How Will We Analyze Lambda Per Method? 

We validate the λ measurement by analyzing the dataset (K=15 directions, orders 1-6) collected during regular training runs. This requires **no additional experiments**—just post-hoc analysis of already-collected data.

### A. Direction Convergence Analysis

**Question**: How many random directions (K) are needed for a stable λ estimate?

**Method** (post-hoc analysis):
- From the K=15 directions collected at each measurement, subsample K ∈ {1, 2, 3, 5, 10, 15}
- For each K, compute λ (using orders 1-6) mean, std, and stderr from the first K directions
- Plot convergence across training: Does optimal K change as training progresses?

**Metrics**:
- **Mean convergence**: Does λ_mean stabilize as K increases? (Plot λ_mean vs K)
- **Standard error decay**: Does stderr ≈ std/√K? (Verify random sampling assumption)
- **Temporal consistency**: Is optimal K the same at epoch 1 vs epoch 50?

**Visualization**:
```
Plot 1: λ_mean ± stderr vs. K at different epochs
Plot 2: Coefficient of variation (std/mean) vs. K across training
Plot 3: Per-method comparison: Do different methods show different direction needs?
```

**Expected outcome**: 
- K=4-7 should be sufficient if landscape is isotropic
- Early training may need higher K due to instability
- Well-regularized methods should converge faster (lower K needed)

---

### B. Derivative Order Sensitivity

**Question**: How many derivative orders are needed for reliable λ estimation?

**Method** (post-hoc analysis):
- From the 1-6 order derivatives collected, recompute λ using different order subsets, keeping K-dirs at 15:
  - **1-2**: Only 1st and 2nd derivatives
  - **1-3**: Up to 3rd order  
  - **1-4**: Up to 4th order
  - **2-4**: Exclude 1st order (test bias 1)
  - **2-5**: Exclude 1st order (test bias 2)
  - **1-6**: All collected orders
- Compare λ estimates across configurations at each measurement

**Metrics**:
- **Order sensitivity**: How much does λ change with additional orders?
- **First-order bias**: Does excluding 1st order shift estimates significantly?
- **Saturation**: Do orders 5-6 add information or just noise?
- **Method dependence**: Do different regularization methods show different order sensitivity?

**Visualization**:
```
Plot 1: λ estimate vs. order configuration (bar plot with error bars)
Plot 2: Per-method comparison: Do different methods show different order needs?
```

**Decision criterion**: 
- Use minimum order where λ estimates differ by < 5% from full 1-6 calculation
- Verify consistency across methods and training phases

**Expected outcome**: 3-4 orders likely sufficient; 6th order may be noisy but validates stability

---

### C. Temporal and Directional Stability (Spaghetti Plots)

**Question**: Do individual directions evolve similarly or do some sharpen while others flatten?

**Method** (post-hoc analysis):
- Plot all K=15 individual direction λ trajectories throughout training
- Overlay mean and confidence intervals
- Compare patterns across regularization methods

**Metrics**:
- **Parallel evolution**: Do directions track together (global smoothing)?
- **Divergence over time**: Does coefficient of variation increase or decrease?
- **Outlier directions**: Are there consistently extreme directions?
- **Method signatures**: Do different regularization strategies show different directional patterns?

**Visualization**:
```
Plot 1: Spaghetti plot - 15 thin colored lines + thick black mean line
        X-axis: Epoch, Y-axis: λ value
Plot 2: Violin plots at key epochs (0, 10, 25, 50) showing distribution evolution
Plot 3: Coefficient of variation (CV = std/mean) vs epoch
Plot 4: Multi-panel comparison across regularization methods
```

**Interpretation**:
- **Parallel lines**: Regularization affects all directions equally (isotropic landscape)
- **Diverging lines**: Loss landscape develops anisotropic structure during training
- **Crossing lines**: Different subspaces sharpen/flatten at different training phases
- **Narrowing distribution**: Training makes landscape more uniform (good)
- **Widening distribution**: Training creates sharp/flat subspaces (potentially problematic)

---

### D. Joint K-Order Optimization

**Question**: What is the optimal combination of K directions and derivative orders for accurate, stable λ estimation?

**Method** (post-hoc analysis):
- For every combination of K ∈ {1, 2, 3, 5, 10, 15} and order_config ∈ {1-2, 1-3, 1-4, 2-4, 2-5, 1-6}:
  - Compute λ estimate using first K directions and specified orders
  - Calculate standard error, coefficient of variation
  - Compare to "ground truth" (K=15, orders 1-6)
- Generate this analysis at key epochs: {0, 5, 10, 25, 50}
- Compare across all regularization methods

**Metrics**:
- **λ estimate error**: |λ(K, order) - λ(15, 1-6)| (lower is better)
- **Standard error**: stderr(K, order)
- **Coefficient of variation**: CV(K, order) = std/mean
- **Stability score**: Combine low error + low variance

**Visualization**:
```
Plot 1: Heatmap - Rows: K values, Columns: Order configs, Color: λ estimate
        Separate heatmaps for each epoch to show temporal evolution
        
Plot 2: Heatmap - Color: Standard error
        Shows measurement uncertainty for each (K, order) combination
        
Plot 3: Heatmap - Color: |λ(K, order) - λ(15, 1-6)| (deviation from "ground truth")
        Reveals which minimal configurations are still accurate
        
Plot 4: Heatmap - Color: Coefficient of variation
        Shows where estimates become unstable
        
Plot 5: Multi-method comparison grid
        Small multiples: Each subplot shows one method's (K, order) landscape
        Reveals if different regularization methods need different measurement configs
        
Plot 6: Pareto frontier plot
        X-axis: Computational cost (K × order)
        Y-axis: Estimation error (lower is better)
        Shows optimal tradeoff points
        
Plot 7: Temporal evolution animation/faceted plot
        Show how optimal (K, order) changes from epochs
        Does early training need more directions/orders than late training?
```

**Decision criterion**:
- Identify minimum (K, order) where:
  - |λ - λ_true| < 0.05 (accurate within 5%)
  - stderr < 0.1 (low measurement uncertainty)
  - CV < 0.2 (stable across directions)
- Check if optimal configuration is consistent across methods and epochs

**Interpretation**:
- **Uniform optimal region**: One (K, order) works for all methods → Use that universally
- **Method-specific optima**: Different regularization needs different measurements → Adaptive strategy
- **Temporal shift**: Early training needs more K/orders → Use adaptive schedule
- **Diminishing returns**: Clear elbow point → Don't oversample

**Expected outcome**: 
- Early training (epoch 0-10): May need K≥7, orders 1-4
- Late training (epoch 40+): K=3-5, orders 1-3 may suffice
- Well-regularized methods (SAM, weight decay): Lower K needed
- Unstable methods (no reg): Higher K needed

---

## Validation Summary

These analyses collectively answer:
1. **Is K=? sufficient?** → Convergence analysis shows where additional directions add no value
2. **Are ? orders needed?** → Order sensitivity shows if ? orders suffice
3. **Is λ isotropic?** → Spaghetti plots reveal directional variance and evolution
4. **Do different methods need different K or orders?** → Cross-method comparisons
5. **What's the optimal (K, order) combination?** → Joint analysis reveals best tradeoff point

**Key advantage**: All analyses use the **same training data**, requiring no additional computational cost beyond storage and post-hoc plotting.

## Interpretation of Possible Results

### Scenario 1: Lambda is a Unified Diagnostic Metric
**Observation**: All well-generalizing methods (regardless of type) converge to λ ∈ [λ_min, λ_max], with low variance.

**Interpretation**: 
- λ measures a fundamental property of the loss landscape related to generalization
- Can be used as a **model-agnostic diagnostic tool**
- Practitioners could monitor λ during training: "Your λ is drifting positive, consider stronger regularization"

**Practical Impact**: HIGH - λ becomes a new tool for hyperparameter tuning and training diagnostics

### Scenario 2: Lambda Differentiates Regularization Types
**Observation**: Different regularization methods cluster at distinct λ values, but within each method, λ correlates with performance.

**Interpretation**:
- Different robustness strategies have different curvature signatures
- Dropout might lead to λ ≈ -2.0, weight decay to λ ≈ -1.5, adversarial training to λ ≈ -4.0
- λ could characterize *how* a model achieves robustness, not just *whether* it's robust

**Practical Impact**: MEDIUM - λ provides insight into regularization mechanism, useful for research

### Scenario 3: Lambda is Method-Specific
**Observation**: λ correlates with performance within a single regularization method but not across methods.

**Interpretation**:
- λ is sensitive to training dynamics in complex ways
- Useful for comparing models trained with the same method
- Not a universal metric for model quality

**Practical Impact**: LOW - Limited applicability, primarily theoretical interest

### Scenario 4: Lambda is Uninformative
**Observation**: λ values are uncorrelated with test accuracy, generalization gap, or calibration across all methods.

**Interpretation**:
- λ may be mathematically well-defined but not practically useful
- High-order derivatives may be too noisy or method-dependent to serve as metrics
- The curvature rate doesn't capture meaningful properties of neural network training

**Practical Impact**: NEGATIVE RESULT - Still publishable; indicates this approach is not viable

## Implementation Details

### Hyperparameter Strategy

**Design Choice**: We use **dataset-specific hyperparameters** to ensure both datasets operate in the "Goldilocks zone" where regularization effects are observable.

#### MNIST Configuration
- **Learning rate**: 0.001
- **Epochs**: 50
- **Batch size**: 128
- **Subset size**: 1000 train / 1000 test (reduced to prevent ceiling effect)
- **LR scheduler**: None
- **Target performance**: Baseline 85-92%, Best methods 93-96%

#### CIFAR10 Configuration  
- **Learning rate**: 0.003 (3x higher than MNIST, balanced to prevent overfitting)
- **Epochs**: 100 (2x longer than MNIST)
- **Batch size**: 128
- **Subset size**: 10000 train / 1000 test (20% of full dataset)
- **LR scheduler**: Cosine annealing (smooth convergence)
- **Target performance**: Baseline 60-70%, Best methods 75-82%

**Rationale**: 
- Different datasets require different training regimes to reach similar **relative performance zones**
- We want both datasets to show:
  - Baseline that has moderate overfitting (5-15% train-test gap)
  - Clear separation between regularization methods (10-15% accuracy range)
  - Room for λ to correlate with performance variation
- **MNIST**: Reduced from 5000→1000 training samples to prevent ceiling effects where all methods achieve 98-99%
- **CIFAR10**: Uses 10,000 samples (20% of dataset) with moderate LR (0.003) to balance learning vs overfitting
  - Previous attempt with 5000 samples + LR=0.01 caused severe overfitting (100% train, 50% test)
  - Current settings target 60-70% baseline test accuracy with visible but not catastrophic overfitting

**Scientific validity**: We're not comparing MNIST vs CIFAR10 performance. We're asking:
- "Within MNIST: Does λ correlate with generalization?"
- "Within CIFAR10: Does λ correlate with generalization?"  
- "Across datasets: Do these correlations show consistent patterns?"

**Optimizer**: Adam for all experiments
**Device**: CUDA (if available)
**Random seed**: 42 for reproducibility

### Lambda Measurement Parameters
- **K_dirs**: 15 random directions per measurement (enables post-hoc convergence analysis)
- **max_order**: 6 (derivatives up to 6th order, enables post-hoc order sensitivity analysis)
- **Measurement frequency**: Adaptive schedule
  - Epochs 0-10: Every epoch
  - Epochs 11-30: Every 2 epochs  
  - Epochs 31-50: Every 5 epochs
  - Epochs 50+: Every 10 epochs
- **Data retention**: Store all 15 individual directions with all derivatives (orders 1-6) at each measurement


### Output Structure

All results are organized in a hierarchical directory structure for easy navigation:

```
results/proof_of_concept/
├── experiment_config.json             # Complete configuration for reproducibility
│
├── 00_summary/                        # High-level overview
│   ├── all_methods_comparison.png     # λ evolution for all methods overlaid
│   ├── performance_vs_lambda.png      # Test accuracy vs λ correlation
│   ├── generalization_gap_vs_lambda.png
│   ├── calibration_vs_lambda.png      # ECE vs λ correlation
│   └── summary_statistics.csv         # Final metrics for all methods
│
├── 01_baseline/                       # No regularization
│   ├── training_metrics.csv           # Epoch-by-epoch train/test acc, loss, ECE
│   ├── lambda_data.npz               # Raw λ data: all 15 directions × all epochs × all orders
│   ├── lambda_evolution.png          # Mean λ ± std over training
│   └── final_model.pt                # Trained model checkpoint
│
├── 02_dropout/                        # Dropout experiments
│   ├── dropout_0.3/
│   │   ├── training_metrics.csv
│   │   ├── lambda_data.npz
│   │   ├── lambda_evolution.png
│   │   └── final_model.pt
│   ├── dropout_0.5/
│   │   └── ...
│   ├── dropout_0.7/
│   │   └── ...
│   └── dropout_comparison.png         # All dropout rates compared
│
├── 03_weight_decay/
│   ├── wd_0.0001/
│   ├── wd_0.001/
│   ├── wd_0.01/
│   └── weight_decay_comparison.png
│
├── 04_data_augmentation/
│   ├── training_metrics.csv
│   ├── lambda_data.npz
│   ├── lambda_evolution.png
│   └── final_model.pt
│
├── 05_label_smoothing/
│   ├── smooth_0.05/
│   ├── smooth_0.1/
│   ├── smooth_0.15/
│   └── label_smoothing_comparison.png
│
├── 06_sam/
│   ├── sam_rho_0.05/
│   ├── sam_rho_0.1/
│   ├── sam_rho_0.2/
│   └── sam_comparison.png
│
├── 07_input_gradient_penalty/
│   ├── igp_0.01/
│   ├── igp_0.1/
│   ├── igp_1.0/
│   └── igp_comparison.png
│
├── validation_analysis/               # Post-hoc λ measurement validation
│   │
│   ├── direction_convergence/
│   │   ├── convergence_epoch_0.png    # λ vs K at epoch 0 (all methods)
│   │   ├── convergence_epoch_10.png
│   │   ├── convergence_epoch_25.png
│   │   ├── convergence_epoch_50.png
│   │   ├── cv_vs_K_across_training.png # Coefficient of variation evolution
│   │   └── per_method_convergence.png  # Small multiples: each method's K needs
│   │
│   ├── order_sensitivity/
│   │   ├── lambda_vs_order_config.png  # Bar plot comparing all order configs
│   │   ├── order_difference_heatmap.png # |λ(order) - λ(1-6)| over training
│   │   ├── per_method_order_sensitivity.png
│   │   └── first_order_bias_analysis.png
│   │
│   ├── temporal_directional_stability/
│   │   ├── spaghetti_baseline.png      # All 15 directions for baseline
│   │   ├── spaghetti_dropout_0.5.png
│   │   ├── spaghetti_weight_decay_0.001.png
│   │   ├── spaghetti_sam_0.1.png
│   │   ├── spaghetti_all_methods_grid.png # Small multiples comparison
│   │   ├── violin_evolution.png        # Distribution at epochs {0, 10, 25, 50}
│   │   ├── cv_over_time.png           # Coefficient of variation vs epoch
│   │   └── direction_correlation_matrix.png # Do directions covary?
│   │
│   └── joint_K_order_optimization/
│       ├── heatmap_lambda_estimate_epoch_0.png    # (K, order) → λ
│       ├── heatmap_lambda_estimate_epoch_10.png
│       ├── heatmap_lambda_estimate_epoch_25.png
│       ├── heatmap_lambda_estimate_epoch_50.png
│       ├── heatmap_standard_error.png             # (K, order) → stderr
│       ├── heatmap_coefficient_variation.png      # (K, order) → CV
│       ├── heatmap_deviation_from_truth.png       # (K, order) → |λ - λ_true|
│       ├── pareto_frontier.png                    # Cost vs accuracy tradeoff
│       ├── per_method_K_order_grid.png           # Small multiples per method
│       ├── temporal_evolution_animation.gif       # How optimal (K,order) changes
│       └── optimal_config_recommendations.txt     # Data-driven recommendations
│
└── cifar10/                           # Same structure repeated for CIFAR-10
    └── ... (identical structure as above)
```

**File Formats**:
- **`.png`**: High-resolution (300 DPI) publication-ready figures
- **`.csv`**: Tabular data for easy loading into Excel/pandas
- **`.npz`**: Compressed NumPy archives with raw λ data arrays
- **`.pt`**: PyTorch model checkpoints
- **`.json`**: Configuration and metadata
- **`.txt`**: Human-readable summaries and recommendations

  
- **`lambda_data.npz`** contains:
  - `directions`: [n_measurements, 15, param_dim] array of sampled directions
  - `derivatives`: [n_measurements, 15, 6] array of derivatives (orders 1-6)
  - `lambda_values`: [n_measurements, 15] array of per-direction λ estimates
  - `epochs`: [n_measurements] array of measurement epochs
  - `train_acc`, `test_acc`, `train_loss`, `test_loss`: [n_measurements] arrays


## Success Criteria

This experiment is successful if we can answer:
1. **Does λ consistently correlate with any known model quality metric?**
2. **Is λ stable (low variance) when it should be?**
3. **Can λ provide actionable insights during training?**

A "yes" to any of these questions establishes λ as a useful tool. A "no" to all three is also valuable—it tells us this particular approach to measuring loss landscape curvature is not the right abstraction for practical use.

## Next Steps After This Experiment

Depending on results:
- **If promising**: Explore regularization
- **If method-specific**: Investigate why different regularization methods have different λ signatures
- **If uninformative**: Explore alternative curvature metrics (Hessian eigenvalues, sharpness measures)
+