# 🗑️ Waste Classification — Progress Tracker & Experiment Log

> **Purpose:** Track all progress, decisions, experiments, and learnings throughout this project.  
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

### Phase 1 — Setup & Dataset Inspection ⬅️ CURRENT
- [x] Create project folder structure with placeholders
- [x] Set up configs/config.yaml with all hyperparameters
- [x] Create requirements.txt, .gitignore, pyproject.toml, README.md
- [x] Create virtual environment and install dependencies
- [x] Verify PyTorch + Torchvision installation (PyTorch 2.14.1+cpu, Torchvision 0.29.1+cpu)
- [ ] Download TrashNet dataset to data/raw/
- [ ] Run data exploration: class counts, sample images
- [ ] Check for corrupt/unreadable images
- [ ] Check for duplicate/near-duplicate images
- [ ] Create stratified train/val/test splits (70/15/15)
- [ ] Save split manifest CSV + class_mapping.json
- [ ] Document dataset findings below

### Phase 2 — Baseline Model
- [ ] Implement Dataset class + DataLoaders
- [ ] Implement transforms (model-specific preprocessing)
- [ ] Build frozen MobileNetV3-Large model
- [ ] Implement training loop (engine.py)
- [ ] Implement validation loop
- [ ] Training with checkpointing
- [ ] Generate training curves
- [ ] Record baseline results

### Phase 3 — Transfer Learning & Fine-Tuning
- [ ] Step A: Train classification head (frozen backbone)
- [ ] Step B: Unfreeze deeper layers, fine-tune with lower LR
- [ ] Step C: Compare frozen vs fine-tuned results
- [ ] Data augmentation (training only)
- [ ] Early stopping + LR scheduling
- [ ] Document improvement rationale

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

## 📊 Experiment Log

*Record every training run here. Never fabricate results — only actual numbers from completed runs.*

| # | Date | Experiment | Epochs | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Test Acc | Test F1 | Notes |
|---|------|-----------|--------|------------|-----------|----------|---------|--------|----------|---------|-------|
| — | — | *No experiments yet* | — | — | — | — | — | — | — | — | — |

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

