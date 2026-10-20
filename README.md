# Intelligent Waste Classification with PyTorch

> A portfolio-quality computer vision project classifying waste images into 6 categories using transfer learning with MobileNetV3.

🚧 **Work in Progress** — See [PROGRESS.md](PROGRESS.md) for milestone checklist and [EXPERIMENTS.md](EXPERIMENTS.md) for detailed progression case studies ($X \rightarrow Y$ performance gains).

## Project Overview

This project builds an image classifier to sort waste into categories:
**cardboard | glass | metal | paper | plastic | trash**

### Key Features (planned)
- Transfer learning with MobileNetV3-Large (pretrained on ImageNet)
- Two-phase training: frozen backbone → fine-tuned layers
- Comprehensive evaluation with per-class metrics and error analysis
- Reusable inference pipeline
- Reproducible experiments with configuration files

## Tech Stack
**PyTorch, Torchvision, OpenCV, MobileNetV3-Large, Transfer Learning, Fine-Tuning, scikit-learn**

- **Framework:** PyTorch + Torchvision
- **Architecture:** MobileNetV3-Large (pretrained on ImageNet)
- **Techniques:** Transfer Learning, Fine-Tuning, Data Augmentation
- **Image Processing:** OpenCV (data quality analysis, visualization), Pillow, Torchvision transforms
- **Dataset:** TrashNet (MIT License, ~2,527 images, 6 classes)
- **Evaluation:** scikit-learn (macro F1, confusion matrix, per-class metrics)
- **Config & Testing:** PyYAML, pytest

## Project Structure
```
waste-classification/
├── configs/config.yaml          # All hyperparameters
├── data/raw/                    # Downloaded dataset (not in git)
├── data/splits/                 # Split manifests
├── notebooks/                   # Exploration notebooks
├── src/waste_classifier/        # Main package
│   ├── data/                    # Dataset + transforms
│   ├── models/                  # Model building
│   ├── training/                # Train loop + engine
│   ├── evaluation/              # Metrics + error analysis
│   ├── inference/               # Prediction pipeline
│   └── utils/                   # Reproducibility + metrics
├── scripts/prepare_data.py      # Data preparation
├── tests/                       # pytest tests
├── artifacts/                   # Saved models + reports
├── PROGRESS.md                  # Experiment log & decisions
└── README.md                    # This file
```

## Setup

```bash
# Create virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Dataset
Using [TrashNet](https://github.com/garythung/trashnet) by Gary Thung & Mindy Yang (MIT License).

## Results

> ⚠️ No results yet — training has not been run. Results will be added with actual metrics only.

| Model | Train Acc | Val Acc | Test Acc | Macro F1 | Notes |
|-------|-----------|---------|----------|----------|-------|
| — | — | — | — | — | *Pending Phase 2+* |

## License
MIT

