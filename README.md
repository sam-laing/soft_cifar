# Soft CIFAR: Exploring Neural Network Training with Soft Labels

This repository contains the implementation and experimental code for our study on the impact of soft labels in neural network training, specifically focusing on the CIFAR-10 dataset.

## Overview

We investigate whether training neural networks with soft labels (probabilistic class distributions from multiple annotators) provides benefits over traditional hard labels. Our research compares several uncertainty-aware network architectures and regularization techniques across multiple robustness metrics.

## Key Features

- Implementation of multiple uncertainty-aware architectures:
  - Spectral Normalized Gaussian Process (SNGP)
  - Deterministic Uncertainty Quantification (DUQ)
  - Monte Carlo Dropout

- Comprehensive evaluation framework measuring:
  - Classification accuracy
  - Out-of-distribution detection (OOD AUROC)
  - Expected Calibration Error (ECE)
  - Adversarial robustness via FGSM attacks

- Dataset handling for both hard and soft CIFAR-10 labels

- Regularization techniques implementation:
  - MixUp
  - CutMix
  - Spectral normalization
  - Gradient penalties

## Results

Our experiments show that:

- Soft labels generally improve model robustness across multiple dimensions
- Different network architectures excel at different robustness metrics
- SNGP performs best at OOD detection when trained with soft labels
- Monte Carlo Dropout provides superior adversarial robustness
- Regularization techniques like MixUp and CutMix cannot effectively replicate the benefits of training with soft labels
- For a more detailed overview, access the pdf of the report located under uncertainty_report/report.pdf

## Model Architecture

We use ResNet architectures of varying depths, with the primary experiments conducted using 20-layer models (~270,000 parameters). Each network is optionally wrapped with uncertainty quantification methods.

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- Weights & Biases for experiment tracking





## Usage

### Basic Training

To train a basic ResNet model with hard labels:

```bash
python train.py --unc_method basic --hard True --do_augmentation True
