# Diffusion-based Controllable Time-step Consistency for Semi-Supervised Medical Image Segmentation

This is the official implementation of "Diffusion-based Controllable Time-step Consistency for Semi-Supervised Medical Image Segmentation" (under review).

## Overview

This repository presents a novel semi-supervised medical image segmentation approach that leverages diffusion models with controllable time-step consistency. The method combines labeled and unlabeled data to improve segmentation performance through a consistency regularization strategy with adaptive time-step selection.

## Key Features

- **Diffusion-based Framework**: Utilizes diffusion models for semi-supervised segmentation
- **Time-step Consistency**: Proposes controllable consistency learning across multiple diffusion time steps
- **Adaptive Time-step Selection**: Automatically selects optimal diffusion steps based on training dynamics

## Repository Structure

```
├── code/                # Source code for training and evaluation
│   ├── dataloaders/     # Data loading utilities
│   ├── guided_diffusion/ # Diffusion model implementation
│   ├── light_training/  # Training utilities
│   ├── module/          # Model components
│   ├── utils/           # Helper functions
├── data/                           # Data directory
├── logs_acdc/                      # checkpoints
├── logs_prostate/                  # checkpoints
└── README.md
```

## Requirements

```
torch >= 1.9.0
torchvision >= 0.10.0
numpy >= 1.19.0
scipy >= 1.5.0
tensorboard >= 2.5.0
monai >= 0.8.0
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd SemiDiff/code

```

## Data Preparation

Our data processing pipeline references the excellent work from [ABD](https://github.com/Star-chy/ABD/tree/main). Please follow their preprocessing guidelines to prepare your data:

## Training

### ACDC Dataset

```bash
python train_acdc_semi.py
```

### Prostate Dataset

```bash
python train_prostate_semi.py 
```

## Inference

### ACDC Dataset
```bash
python evaluate_acdc.py
```
### Prostate Dataset

```bash
python evaluate_prostate.py
```


## Acknowledgments

Our code is largely based on the following excellent works:

- [SSL4MIS](https://github.com/UESTC-Med424-JYX/SSL4MIS) - Semi-supervised learning for medical image segmentation
- [BCP](https://github.com/UESTC-Med424-JYX/BCP) - Boundary-aware contrastive learning
- [ABD](https://github.com/Star-chy/ABD) - Data preprocessing and augmentation strategies
- [Diff-SFCT](https://github.com/UESTC-Med424-JYX/Diff-SFCT) - Diffusion-based segmentation framework
- [SCP-Net](https://github.com/UESTC-Med424-JYX/SCP-Net) - Semi-supervised segmentation with contrastive learning

We sincerely thank the authors for their valuable contributions to medical image segmentation research.

## Citation

If you find this work useful, please cite:

```bibtex
@article{SemiDiff2024,
  title={Diffusion-based Controllable Time-step Consistency for Semi-Supervised Medical Image Segmentation},
  author={...},
  journal={...},
  year={2024},
  note={Under review}
}
```
