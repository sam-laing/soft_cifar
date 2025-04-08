# Do Soft Labels Enhance Neural Network Training?

## Abstract  
We investigate training ResNet variants on CIFAR-10 with soft labels versus hard labels, evaluating accuracy, calibration, OOD detection, and adversarial robustness. Our experiments compare:
- Basic ResNet
- SNGP (Spectral Normalized Gaussian Process)  
- DUQ (Deterministic Uncertainty Quantification)
- MC Dropout (Monte Carlo Dropout)

**Key Findings**:
1. Soft labels improve robustness (+1.5% accuracy, +15% OOD detection)
2. MC Dropout provides best adversarial defense
3. SNGP excels at uncertainty quantification
4. Regularization using Mixup and Cutmix cannot fully replicate soft label benefits

## Methods

### Architectures
| Model     | Parameters | Key Features |
| --------- | ---------- | ------------ |
| ResNet-20 | 272K       | Baseline     |
| +SNGP     | 349K       | GP layer     |
| +DUQ      | 928K       | Centroids    |
| +Dropout  | 272K       | p=0.3        |

### Training
- **Optimizer**: SGD (lr=0.05, momentum=0.9)
- **Schedule**: Warmup + MultiStepLR  
- **Epochs**: 200
- **Batch Size**: 128

### Metrics
1. **ECE**: Calibration error across 10 bins
2. **OOD AUROC**: SVHN vs CIFAR-10
3. **FGSM Robustness**: ε=0.005 attack
4. **Accuracy**: Standard classification

## Results

| Model Type       | Accuracy         | OOD AUROC         | ECE               | Attack Decrease   |
| ---------------- | ---------------- | ----------------- | ----------------- | ----------------- |
| **Basic Models** |                  |                   |                   |                   |
| soft labels      | 82.92 ± 0.84     | 0.775 ± 0.043     | 0.090 ± 0.003     | **0.352 ± 0.015** |
| hard labels      | 81.37 ± 0.60     | 0.711 ± 0.054     | **0.046 ± 0.003** | 0.441 ± 0.018     |
| soft + Mixup     | 82.15 ± 1.42     | **0.785 ± 0.055** | 0.103 ± 0.007     | 0.356 ± 0.017     |
| hard + Mixup     | 81.65 ± 0.58     | 0.682 ± 0.076     | 0.070 ± 0.003     | 0.445 ± 0.014     |
| soft + CutMix    | **83.52 ± 1.40** | 0.748 ± 0.079     | 0.108 ± 0.003     | 0.389 ± 0.010     |
| hard + CutMix    | 82.68 ± 0.50     | 0.527 ± 0.067     | 0.082 ± 0.003     | 0.465 ± 0.021     |

| **Dropout (No Aug)**                 |                   |                  |                  |                  |
| soft labels                          | 78.19 ± 0.66      | 0.658 ± 0.067    | 0.111 ± 0.004    | **0.175 ± 0.008**|
| hard labels                          | 77.65 ± 0.96      | 0.588 ± 0.081    | **0.057 ± 0.002**| 0.203 ± 0.008    |
| soft + Mixup                         | 78.79 ± 0.69      | **0.689 ± 0.063**| 0.126 ± 0.003    | 0.176 ± 0.013    |
| hard + Mixup                         | 78.15 ± 0.88      | 0.532 ± 0.099    | 0.088 ± 0.003    | 0.193 ± 0.009    |
| soft + CutMix                        | **80.15 ± 0.66**  | 0.659 ± 0.075    | 0.132 ± 0.007    | 0.177 ± 0.006    |
| hard + CutMix                        | 78.56 ± 0.72      | 0.443 ± 0.086    | 0.095 ± 0.003    | 0.195 ± 0.007    |

| **Dropout (With Aug)**               |                   |                  |                  |                  |
| soft labels                          | 82.93 ± 0.14      | **0.785 ± 0.037**| 0.100 ± 0.001    | 0.152 ± 0.004    |
| hard labels                          | 82.40 ± 0.54      | 0.548 ± 0.026    | **0.070 ± 0.001**| 0.169 ± 0.001    |
| soft + Mixup                         | 82.44 ± 0.85      | 0.734 ± 0.018    | 0.110 ± 0.004    | **0.141 ± 0.006**|
| hard + Mixup                         | 82.18 ± 0.36      | 0.636 ± 0.039    | 0.088 ± 0.004    | 0.175 ± 0.008    |
| soft + CutMix                        | **83.38 ± 0.94**  | 0.761 ± 0.038    | 0.117 ± 0.001    | 0.158 ± 0.005    |
| hard + CutMix                        | 82.71 ± 1.05      | 0.558 ± 0.027    | 0.094 ± 0.002    | 0.173 ± 0.008    |

| **DUQ Models**                       |                   |                  |                  |                  |
| soft labels                          | 81.91 ± 1.20      | **0.797 ± 0.076**| 0.693 ± 0.011    | 0.422 ± 0.021    |
| hard labels                          | 81.65 ± 1.03      | 0.667 ± 0.074    | 0.692 ± 0.008    | 0.427 ± 0.027    |
| soft + Mixup                         | **82.00 ± 1.08**  | 0.714 ± 0.020    | 0.694 ± 0.009    | **0.410 ± 0.014**|
| hard + Mixup                         | 80.91 ± 1.22      | 0.604 ± 0.034    | **0.685 ± 0.010**| 0.410 ± 0.018    |
| soft + CutMix                        | 81.92 ± 1.57      | 0.717 ± 0.029    | 0.692 ± 0.014    | 0.434 ± 0.017    |
| hard + CutMix                        | 81.59 ± 1.67      | 0.617 ± 0.078    | 0.690 ± 0.015    | 0.430 ± 0.019    |

| **SNGP Models** |      |      |      |      |
| --------------- | ---- | ---- | ---- | ---- |
|                 |      |      |      |      |
| soft labels | 80.80 ± 1.18 | **0.844 ± 0.032** | 0.089 ± 0.004 | **0.356 ± 0.008** |
| ----------- | ------------ | ----------------- | ------------- | ----------------- |
|             |              |                   |               |                   |
| hard labels | 80.67 ± 1.30 | 0.779 ± 0.058 | **0.047 ± 0.003** | 0.443 ± 0.013 |
| ----------- | ------------ | ------------- | ----------------- | ------------- |
|             |              |               |                   |               |
| soft + Mixup | **81.08 ± 0.80** | 0.743 ± 0.069 | 0.103 ± 0.005 | 0.362 ± 0.012 |
| ------------ | ---------------- | ------------- | ------------- | ------------- |
|              |                  |               |               |               |
| hard + Mixup | 80.89 ± 0.60 | 0.670 ± 0.071 | 0.069 ± 0.003 | 0.455 ± 0.013 |
| ------------ | ------------ | ------------- | ------------- | ------------- |
|              |              |               |               |               |
| soft + CutMix | 80.95 ± 0.93 | 0.752 ± 0.063 | 0.103 ± 0.002 | 0.382 ± 0.013 |
| ------------- | ------------ | ------------- | ------------- | ------------- |
|               |              |               |               |               |
| hard + CutMix | 80.85 ± 0.70 | 0.592 ± 0.062 | 0.073 ± 0.003 | 0.462 ± 0.019 |
| ------------- | ------------ | ------------- | ------------- | ------------- |
|               |              |               |               |               |







## Conclusion
Soft labels provide consistent robustness benefits across architectures, with MC Dropout being particularly effective for adversarial defense and SNGP for uncertainty quantification.