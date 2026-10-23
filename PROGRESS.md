# 🗑️ Waste Classification — Progress Tracker & Experiment Log

> **Purpose:** Track all progress, decisions, experiments, and learnings throughout this project.  
> **Evolution Log:** See [EXPERIMENTS.md](EXPERIMENTS.md) for the detailed step-by-step case studies tracking transitions from $X \rightarrow Y$ and quantitative improvements.  
> **Updated:** 2026-10-20

---

## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Framework** | PyTorch + Torchvision |
| **Architecture** | MobileNetV3-Large (primary) |
| **Dataset** | TrashNet — MIT License, ~2,527 images, 6 classes |
| **Classes** | cardboard, glass, metal, paper, plastic, trash |
| **Goal** | Portfolio-quality classifier with proper evaluation for AI Engineer resume |

---

## 🗺️ Phase Checklist

### Phase 1 — Setup & Dataset Inspection ✅ COMPLETED
- [x] Create project folder structure with placeholders
- [x] Set up configs/config.yaml with all hyperparameters
- [x] Create requirements.txt, .gitignore, pyproject.toml, README.md
- [x] Create virtual environment and install dependencies
- [x] Verify PyTorch + Torchvision installation (PyTorch 2.14.1+cpu, Torchvision 0.29.1+cpu)
- [x] Download TrashNet dataset to data/raw/ (2,527 images, 6 classes, MIT license)
- [x] Run data exploration: class counts, sample images, resolutions
- [x] Check for corrupt/unreadable images (0 corrupt files found)
- [x] Check for duplicate/near-duplicate images (8 perceptual dHash clusters detected)
- [x] Create stratified train/val/test splits (70/15/15: 1768 train, 379 val, 380 test)
- [x] Save split manifest CSV + class_mapping.json
- [x] Document dataset findings below

### Phase 2 — Baseline Model ✅ COMPLETED
- [x] Implement Dataset class + DataLoaders (`src/waste_classifier/data/dataset.py`)
- [x] Implement transforms with model-specific preprocessing (`src/waste_classifier/data/transforms.py`)
- [x] Build frozen MobileNetV3-Large model (`src/waste_classifier/models/build_model.py`)
- [x] Implement training loop with autograd explanation (`src/waste_classifier/training/engine.py`)
- [x] Implement validation loop with torch.no_grad()
- [x] Train baseline model on GPU (NVIDIA GTX 1650, 10 epochs, 3.3 min)
- [x] Checkpoint best model (`artifacts/best_model.pth`)
- [x] Generate training curves (`artifacts/training_curves.png`)
- [x] Record baseline results in EXPERIMENTS.md (Val Acc: 84.17%, Val F1: 0.8194)

### Phase 3 — Transfer Learning & Fine-Tuning ⬅️ CURRENT
- [x] Step A: Train classification head (frozen backbone) — COMPLETED in Phase 2 (84.17% val acc)
- [ ] Step B: Unfreeze deeper layers (last 4 blocks), fine-tune with lower LR (1e-5)
- [ ] Step C: Compare frozen vs fine-tuned results
- [ ] Data augmentation verification
- [ ] Early stopping + LR scheduling
- [ ] Document improvement rationale in EXPERIMENTS.md

### Phase 4 — Evaluation & Error Analysis
- [ ] Test set evaluation (accuracy, macro F1)
- [ ] Per-class precision/recall/F1 table
- [ ] Confusion matrix visualization
- [ ] Misclassified image examples
- [ ] Error pattern analysis (which classes confused? why?)
- [ ] Model comparison table

### Phase 5 — Inference Pipeline
- [ ] predict.py with single-image inference
- [ ] Load checkpoint without retraining
- [ ] Top-3 predictions with confidence
- [ ] Unit tests (test_dataset.py, test_inference.py)
- [ ] Fresh process verification

### Phase 6 — Polish & Deployment
- [ ] Professional README with actual metrics
- [ ] Dataset attribution and license
- [ ] Clean repository
- [ ] Optional: FastAPI /predict endpoint
- [ ] Optional: Docker

---

## 🔍 Phase 1 Dataset Findings & Data Quality Report

| Metric | Result | Interview Note |
|---|---|---|
| **Total Images** | 2,527 | Clean, focused dataset (~41 MB resized). |
| **Corrupt Files** | 0 | All files verified via PIL verify() and OpenCV imread(). |
| **Image Resolution** | 512 × 384 × 3 (RGB) | Aspect ratio 1.333 (4:3 landscape) across all images. |
| **Duplicates (dHash)** | 8 clusters | Plain white backgrounds or near-identical multi-shots. Documented for leakage control. |

### Class Breakdown & Imbalance
| Class | Total | % of Dataset | Train (70%) | Val (15%) | Test (15%) |
|---|---|---|---|---|---|
| **paper** | 594 | 23.5% | 416 | 89 | 89 |
| **glass** | 501 | 19.8% | 350 | 75 | 76 |
| **plastic** | 482 | 19.1% | 337 | 72 | 73 |
| **metal** | 410 | 16.2% | 287 | 61 | 62 |
| **cardboard** | 403 | 15.9% | 282 | 61 | 60 |
| **trash** | 137 | **5.4%** | 96 | 21 | 20 |
| **Total** | **2,527** | **100.0%** | **1,768** | **379** | **380** |

> **Key Observation:** `trash` is heavily under-represented (only 5.4% of the dataset, vs 23.5% for `paper`). This makes **macro F1** and **per-class recall** critical evaluation metrics for Phase 4, as a naive model could score high accuracy while completely failing on the minority `trash` class.

---

## 📊 Experiment Log

*Record every training run here. Never fabricate results — only actual numbers from completed runs.*

| # | Date | Experiment | Epochs | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Test Acc | Test F1 | Notes |
|---|------|-----------|--------|------------|-----------|----------|---------|--------|----------|---------|-------|
| 1 | 2026-10-24 | Frozen MobileNetV3-Large Head | 10 (early stop) | 0.3238 | 88.86% | 0.4986 | **84.17%** | **0.8194** | *Pending Ph4* | *Pending Ph4* | Best at epoch 6. Adam lr=1e-3, ReduceLROnPlateau. Trained in 3.3 min on GTX 1650. |

---

## 🔑 Key Decisions & Rationale

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-10-20 | Project created | Personal project for AI Engineer resume |
| 2026-10-20 | PyTorch over TF/Keras | Industry standard, better for portfolio, interview discussions |
| 2026-10-20 | MobileNetV3-Large as primary | Modern, efficient, good accuracy/speed tradeoff vs VGG16/ResNet50 |
| 2026-10-20 | TrashNet dataset (MIT) | Well-known, 6 classes, small enough for fast iteration, clear license |
| 2026-10-20 | 70/15/15 stratified splits | Standard ratio; stratified to handle class imbalance (Trash has ~137 vs Paper ~594) |

---

## 📝 Learnings from Reference Project

**Reference:** `github.com/shouzafzafar/waste-classification` (VGG16 + ResNet50, TF/Keras)

### What they did well:
- Compared two architectures (VGG16 vs ResNet50)
- Documented accuracy progression (43% → 80.8%)
- Used callbacks (early stopping, LR reduction, checkpointing)
- Two-phase training: freeze → fine-tune
- Dropout (0.5) for regularization

### Their accuracy progression explained:
| Accuracy | Likely Cause |
|----------|-------------|
| **43%** | Wrong preprocessing (`1/255` instead of model-specific) or fully frozen backbone |
| **~70-75%** | Correct preprocessing + frozen backbone + custom head |
| **80.8%** | Fine-tuning unfrozen upper layers with low LR (~1e-5) |
| **90%+** | Needs: class weighting + ReduceLROnPlateau + modern architecture |

### Common pitfalls we're avoiding:
- ❌ Wrong preprocessing → ✅ Using torchvision pretrained weights API (handles normalization automatically)
- ❌ No class weighting on imbalanced data → ✅ Will evaluate and decide based on evidence
- ❌ Random splits → ✅ Stratified splits to preserve class distribution
- ❌ No error analysis → ✅ Full Phase 4 with confusion matrix + misclassified examples
- ❌ Evaluating on val set as "test" → ✅ Strict train/val/test separation

### What we're doing differently:
- PyTorch instead of TF/Keras
- MobileNetV3-Large instead of VGG16/ResNet50 (more modern, efficient)
- Model-specific preprocessing via Torchvision pretrained weights enum
- Comprehensive error analysis (macro F1, per-class, confusion matrix)
- Configuration file (YAML) instead of hardcoded constants
- Reproducible seeds + deterministic mode
- Reusable inference pipeline with tests

---

## ⚠️ Ground Rules
1. **Never fabricate metrics** — only report numbers from actual completed runs
2. **Validation for model selection, test for final evaluation** — no peeking
3. **Explain every design decision** — this is interview prep
4. **Evidence-based changes** — don't add class weights/augmentation "just because"
5. **Document as we go** — update this file after every significant step

