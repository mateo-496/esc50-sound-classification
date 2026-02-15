---
title: ESC50 Audio Classifier
emoji: 🔊
colorFrom: gray
colorTo: purple
sdk: docker
pinned: false
---

# Environmental Sound Classification (ESC50) with Deep CNNs

A PyTorch reimplementation of the deep convolutional neural network approach from [Salamon & Bello (2017)](https://arxiv.org/pdf/1608.04363) for environmental sound classification, extended to handle 50 classes instead of the original 10.

## Overview

This project implements a deep CNN architecture for environmental sound classification using log-mel spectrograms as input features. The implementation follows the methodology described in the paper "Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification" but is scaled to work with a more challenging 50-class classification task.

### Key Features

- Deep CNN with 3 convolutional layers + 2 fully connected layers
- Log-mel spectrogram feature extraction using Essentia
- Data augmentation (time stretching, pitch shifting, dynamic range compression)
- Overlapping patch prediction for validation (1-frame hop)

## Results

| Dataset | Classes | Accuracy (with augmentation) |
|---------|---------|--------------------------------|
| UrbanSound8K (paper) | 10 | 79% |
| **This project** | **50** | **74%** |

## Architecture

### Model Structure

```
Input: Log-mel Spectrogram (128 × 128)
    ↓
Conv2D(1→24, 5×5) + ReLU + MaxPool(4×2)
    ↓
Conv2D(24→48, 5×5) + ReLU + MaxPool(4×2)
    ↓
Conv2D(48→48, 5×5) + ReLU
    ↓
Flatten → Dense(2400→64) + ReLU + Dropout(0.5)
    ↓
Dense(64→50) + Softmax
    ↓
Output: 50 classes
```

### Training Configuration

- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.01
- **Batch Size**: 100 TF-patches
- **L2 Regularization**: 0.001 (on classifier layers only)
- **Dropout**: 0.5 (on classifier layers)
- **Epochs**: 100