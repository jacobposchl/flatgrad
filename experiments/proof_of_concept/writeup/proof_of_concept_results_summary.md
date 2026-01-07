# Proof of Concept: Results Summary

## Experimental Setup

### Datasets
- **CIFAR10**: 50,000 training images, 10,000 test images (full dataset used)
- **MNIST**: 60,000 training images, 10,000 test images

### Model Architectures
- **CIFAR10**: ConvNet (4 conv layers, 2 FC layers, ~1.5M parameters)
- **MNIST**: MLP (4 FC layers, ~2.6M parameters)

### Training Configuration
- **Epochs**: 100 (CIFAR10), 120 (MNIST)
- **Batch Size**: 128
- **Optimizer**: Adam (lr=0.001) or SGD for SAM
- **LR Scheduler**: Cosine annealing (CIFAR10 only)

### Regularization Methods Tested (17 total)
1. Baseline (no regularization)
2. Dropout (0.3, 0.5, 0.7)
3. Weight Decay (0.0001, 0.001, 0.01)
4. Data Augmentation
5. Label Smoothing (0.05, 0.1, 0.15)
6. SAM - Sharpness Aware Minimization (ρ=0.05, 0.1, 0.2)
7. IGP - Input Gradient Penalty (scale=0.01, 0.1, 1.0)

### Lambda Measurement Parameters
- **Directions per measurement**: 15 random unit directions
- **Derivative order**: Up to 6th order
- **Measurement schedule**: Adaptive (dense early training, sparse later)
- **Method**: Log-linear regression on derivative ratios per direction

### Performance Metrics Collected
- Test Accuracy
- Train Accuracy
- Generalization Gap (Train - Test Accuracy)
- ECE (Expected Calibration Error, 10 bins)
- Lambda Mean (average across directions)
- Lambda Std (standard deviation across directions)

---

## CIFAR10 Results

### Performance Statistics (N=17 methods)
- **Test Accuracy**: Range [0.7784, 0.8893], Mean 0.8145 ± 0.0222
- **Train Accuracy**: Range [0.8638, 1.0000], Mean 0.9983 ± 0.0341
- **Generalization Gap**: Range [0.0854, 0.2216], Mean 0.1838 ± 0.0277
- **ECE**: Range [0.0091, 0.2317], Mean 0.1276 ± 0.0479

### Lambda Statistics (N=17 methods)
- **Lambda Mean**: Range [-2.4235, -0.2985], Mean -0.9493 ± 0.6432
- **Lambda Std**: Range [0.0224, 0.0674], Mean 0.0504 ± 0.0128
- **Lambda Min**: Range [-2.5937, -0.3699], Mean -1.0848 ± 0.6766
- **Lambda Max**: Range [-2.1776, -0.1987], Mean -0.7971 ± 0.6069
- **Lambda Range**: Range [0.0904, 0.2348], Mean 0.1738 ± 0.0353
- **Lambda Skewness**: Range [-0.6894, 1.5018], Mean 0.4186 ± 0.5618
- **Lambda Kurtosis**: Range [-1.1895, 2.6458], Mean 0.3421 ± 1.0383

### Correlations: Lambda Mean vs Performance Metrics
| Performance Metric | Pearson r | p-value | Significant (p<0.05) |
|-------------------|-----------|---------|---------------------|
| Test Accuracy | 0.0966 | 0.7124 | No |
| Generalization Gap | -0.0863 | 0.7419 | No |
| ECE (Calibration) | -0.3549 | 0.1622 | No |
| Train Accuracy | -0.0405 | 0.8772 | No |

### Correlations: Lambda Std vs Performance Metrics
| Performance Metric | Pearson r | Spearman r | p-value (Pearson) | Significant |
|-------------------|-----------|------------|-------------------|-------------|
| Test Accuracy | 0.3730 | 0.6471 | 0.1403 | No |
| Generalization Gap | -0.3509 | -0.6863 | 0.1673 | No |
| ECE (Calibration) | -0.2340 | 0.0000 | 0.3660 | No |
| Train Accuracy | -0.2363 | -0.6104 | 0.3612 | No |

### Correlations: Lambda Skewness vs Performance Metrics
| Performance Metric | Pearson r | p-value | Significant |
|-------------------|-----------|---------|-------------|
| Test Accuracy | 0.6325 | 0.0064 | **Yes** |
| Generalization Gap | -0.6666 | 0.0035 | **Yes** |
| ECE (Calibration) | 0.1961 | 0.4507 | No |
| Train Accuracy | -0.7260 | 0.0010 | **Yes** |

### Correlations: Lambda Kurtosis vs Performance Metrics
| Performance Metric | Pearson r | p-value | Significant |
|-------------------|-----------|---------|-------------|
| Test Accuracy | 0.5073 | 0.0377 | **Yes** |
| Generalization Gap | -0.5654 | 0.0180 | **Yes** |
| ECE (Calibration) | -0.4889 | 0.0464 | **Yes** |
| Train Accuracy | -0.7225 | 0.0011 | **Yes** |

### Correlations: Lambda Range vs Performance Metrics
| Performance Metric | Pearson r | p-value | Significant |
|-------------------|-----------|---------|-------------|
| Test Accuracy | 0.4086 | 0.1035 | No |
| Generalization Gap | -0.4128 | 0.0996 | No |
| ECE (Calibration) | -0.3779 | 0.1348 | No |
| Train Accuracy | -0.3882 | 0.1237 | No |

### Top 5 Methods by Test Accuracy
1. Augmentation: 0.8893, λ_mean=-0.9232, λ_skew=0.7598, ECE=0.0544
2. Dropout 0.7: 0.8222, λ_mean=-0.2985, λ_skew=0.0896, ECE=0.1355
3. SAM 0.1: 0.8195, λ_mean=-0.8825, λ_skew=0.5612, ECE=0.1206
4. SAM 0.2: 0.8185, λ_mean=-1.0580, λ_skew=0.7887, ECE=0.1163
5. IGP 0.01: 0.8181, λ_mean=-0.5293, λ_skew=-0.2197, ECE=0.1391

### Top 5 Methods by Calibration (Lowest ECE)
1. Weight Decay 0.01: ECE=0.0091, λ_mean=-1.0518, λ_skew=1.0176, Acc=0.7942
2. Augmentation: ECE=0.0544, λ_mean=-0.9232, λ_skew=0.7598, Acc=0.8893
3. Weight Decay 0.001: ECE=0.0898, λ_mean=-0.7247, λ_skew=0.9267, Acc=0.7784
4. Weight Decay 0.0001: ECE=0.1143, λ_mean=-0.8237, λ_skew=1.0095, Acc=0.7974
5. SAM 0.2: ECE=0.1163, λ_mean=-1.0580, λ_skew=0.7887, Acc=0.8185

### Top 5 Methods by Flatness (Most Negative Lambda Mean)
1. Label Smoothing 0.15: λ_mean=-2.4235, λ_skew=-0.4085, ECE=0.2317, Acc=0.8069
2. Label Smoothing 0.1: λ_mean=-2.2246, λ_skew=-0.6894, ECE=0.1845, Acc=0.8126
3. Label Smoothing 0.05: λ_mean=-1.9550, λ_skew=-0.1958, ECE=0.1243, Acc=0.8047
4. SAM 0.2: λ_mean=-1.0580, λ_skew=0.7887, ECE=0.1163, Acc=0.8185
5. Weight Decay 0.01: λ_mean=-1.0518, λ_skew=1.0176, ECE=0.0091, Acc=0.7942

---

## Temporal Dynamics (CIFAR10)

### Lambda Evolution During Training
- **Early Training (Epoch 0-10)**: Mean λ = -2.8133
- **Mid Training (Epoch 20-40)**: Mean λ = -1.2996
- **Late Training (Epoch 80-100)**: Mean λ = -0.9529
- **Average Change**: +1.8603 (from early to late)
- **Direction**: Lambda becomes less negative (sharper) during training

---

## Statistical Hypothesis Tests (CIFAR10)

### T-Test: High vs Low Skewness Methods
- **High Skewness (N=9)**: Gen Gap = 0.1808 ± 0.0380
- **Low Skewness (N=8)**: Gen Gap = 0.1872 ± 0.0093
- **T-statistic**: -0.4598
- **p-value**: 0.6523

### KS Test: ECE Distribution Difference
- **KS Statistic**: 0.2222
- **p-value**: 0.9380

### ANOVA: Lambda Range Across Regularization Types
- **F-statistic**: 9.1625
- **p-value**: 0.0014 (significant)

**Lambda Range by Regularization Type:**
| Type | Lambda Range | N |
|------|-------------|---|
| Label Smoothing | 0.0904 ± 0.0272 | 3 |
| Weight Decay | 0.1643 ± 0.0148 | 3 |
| IGP | 0.1709 ± 0.0136 | 3 |
| SAM | 0.1803 ± 0.0214 | 3 |
| Dropout | 0.2026 ± 0.0333 | 3 |
| Baseline | 0.2241 | 1 |
| Augmentation | 0.2348 | 1 |

### Permutation Test: Lambda Skewness vs Generalization Gap
- **Observed Correlation**: -0.6666
- **Permutation p-value**: 0.0301 (N=10,000 permutations)

---

## Method Clustering (CIFAR10)

### PCA on Lambda Features
- **PC1 Variance Explained**: 56.0%
- **PC2 Variance Explained**: 31.9%
- **Total (PC1+PC2)**: 87.9%

**Features used:** lambda_mean, lambda_std, lambda_range, lambda_skewness, lambda_kurtosis, lambda_cv

---

## Summary of Significant Correlations (p<0.05)

### CIFAR10 (N=17)
**Lambda Skewness:**
- Test Accuracy: r=0.6325, p=0.0064
- Generalization Gap: r=-0.6666, p=0.0035
- Train Accuracy: r=-0.7260, p=0.0010

**Lambda Kurtosis:**
- Test Accuracy: r=0.5073, p=0.0377
- Generalization Gap: r=-0.5654, p=0.0180
- ECE: r=-0.4889, p=0.0464
- Train Accuracy: r=-0.7225, p=0.0011

**Lambda Std (Spearman only):**
- Test Accuracy: ρ=0.6471, p=0.0050
- Generalization Gap: ρ=-0.6863, p=0.0023
- Train Accuracy: ρ=-0.6104, p=0.0093

---

## Files Generated

### Data Files
- `results/proof_of_concept/analysis/all_results.csv` - All method results
- `results/proof_of_concept/analysis/correlations.csv` - Correlation matrix
- `results/proof_of_concept/analysis/directional_correlations.csv` - Extended correlations
- `results/proof_of_concept/analysis/cifar10_directional_statistics.csv` - Per-method lambda statistics

### Visualizations
- `cifar10_correlation_heatmap.png` - Lambda mean/std vs performance
- `cifar10_scatter_plots.png` - Lambda mean vs all metrics
- `cifar10_method_comparison.png` - Methods ranked by lambda
- `cifar10_directional_heatmap.png` - All directional features vs performance
- `cifar10_lambda_distributions.png` - Lambda histograms per method
- `cifar10_directional_scatter.png` - Key directional features
- `cifar10_temporal_dynamics.png` - Lambda evolution over training
- `cifar10_method_clustering_dendrogram.png` - Hierarchical clustering
- `cifar10_lambda_pca.png` - PCA visualization colored by ECE

---

**Date Generated**: January 7, 2026  
**Total Methods Tested**: 17 (CIFAR10)  
**Total Measurements**: 15 directions × 6 derivative orders × ~20 epochs = ~1,800 lambda estimates per method
