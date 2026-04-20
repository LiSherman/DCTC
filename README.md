# SemiDiff

This repository contains the official implementation of the paper:

**"Revealing the Potential of the Diffusion Model in Semi-Supervised Medical Image Segmentation via Controllable Time-Step Consistency"**

## Overview

SemiDiff is a framework designed for semi-supervised medical image segmentation. It leverages the power of diffusion models to achieve controllable time-step consistency, improving segmentation performance with limited labeled data.

## Features

- **Semi-Supervised Learning**: Effectively utilizes both labeled and unlabeled data.
- **Diffusion Models**: Incorporates diffusion models for robust feature learning.
- **Medical Image Segmentation**: Tailored for datasets like ACDC and Prostate.

## Datasets

The following datasets are supported and should be placed in the `./data` directory:

- **ACDC**: Automated Cardiac Diagnosis Challenge dataset.
- **Prostate**: Prostate segmentation dataset.

Please refer to the respective dataset licenses for usage terms.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/SemiDiff.git
   cd SemiDiff
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model on the ACDC dataset:
```bash
python code/train_acdc_semi.py
```

To train the model on the Prostate dataset:
```bash
python code/train_prostate_semi.py
```

### Evaluation

To evaluate the model on the ACDC dataset:
```bash
python code/evaluate_acdc.py
```

To evaluate the model on the Prostate dataset:
```bash
python code/evaluate_prostate.py
```

## Directory Structure

```
SemiDiff/
├── code/                # Source code for training and evaluation
│   ├── dataloaders/     # Data loading utilities
│   ├── guided_diffusion/ # Diffusion model implementation
│   ├── light_training/  # Training utilities
│   ├── module/          # Model components
│   ├── utils/           # Helper functions
├── data/                # Dataset directory (not included in the repository)
│   ├── ACDC/            # ACDC dataset
│   ├── Prostate/        # Prostate dataset
├── logs_acdc/           # Logs for ACDC experiments
├── logs_prostate/       # Logs for Prostate experiments
```

## Citation

If you find this repository useful, please cite our paper:
```
@article{your_paper,
  title={Revealing the Potential of the Diffusion Model in Semi-Supervised Medical Image Segmentation via Controllable Time-Step Consistency},
  author={Your Name and Others},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

We thank the authors of the following repositories for their inspiring work:
- [Guided Diffusion](https://github.com/openai/guided-diffusion)
- [ACDC Dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- [Prostate Dataset](https://www.example.com/prostate-dataset)