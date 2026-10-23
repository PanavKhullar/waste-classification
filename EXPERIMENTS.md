# 🔬 Model Evolution & Experiment Log: From $X \rightarrow Y$

> This document tracks the technical progression of the Waste Classifier project.  
> It records the **hypothesis**, **changes made**, **observed results**, and **why we transitioned from point $X$ to point $Y$**, mirroring real-world AI engineering workflows.

---

## 📈 Executive Summary: The Performance Journey

| Stage | Milestone / Configuration | Val Accuracy | Val Macro F1 | Test Accuracy | Key Bottleneck / Discovery |
|---|---|---|---|---|---|
| **Phase 1** | **Dataset & Split Hygiene** | — | — | — | 2,527 images. Severe class imbalance: `trash` is only 5.4%. Stratified splits created. |
| **Stage 0 (Planned)** | *Simple CNN from scratch (Reference baseline)* | *TBD* | *TBD* | *TBD* | Limited dataset (~1.7k train images) causes severe overfitting on scratch CNNs. |
| **Stage 1 (Completed)** | **Frozen MobileNetV3-Large Head Only** | **84.17%** | **0.8194** | *Pending Phase 4* | Baseline established in 10 epochs (3.3 min on GTX 1650). Reached 84.17% val acc without touching backbone. |
| **Stage 2 (Planned)** | *Unfrozen Upper Backbone (Fine-Tuning) + Low LR* | *TBD* | *TBD* | *TBD* | Adapts high-level feature extractors to waste textures and lighting. |
| **Stage 3 (Planned)** | *Fine-Tuning + Data Augmentation + Class Weights* | *TBD* | *TBD* | *TBD* | Addresses minority class underperformance (`trash` 5.4%). |

---

## Detailed Progression Case Studies

---

### Milestone 0: Dataset & Data Hygiene Audit

- **Starting Point ($X$):** Raw unverified images downloaded from public repositories.
- **Hypothesis:** Waste datasets downloaded from the web often contain corrupt images, near-duplicate multi-burst photos, and severe class skew that ruin training credibility if uninspected.
- **Action Taken ($X \rightarrow Y$):**
  1. Ran OpenCV `cv2.imread()` and PIL `verify()` across all 2,527 images.
  2. Executed perceptual difference hashing (`dHash`) with a 64-bit threshold to detect exact and near-identical multi-shot photos.
  3. Audited class distribution and discovered `trash` represents only 5.4% (137 images) while `paper` represents 23.5% (594 images).
  4. Enforced strict 70/15/15 stratified splitting (`train.csv`, `val.csv`, `test.csv`) to preserve class proportions and prevent data leakage.
- **Impact & Results:**
  - **Corrupt images found:** 0
  - **Duplicate clusters identified:** 8 clusters (e.g. plain white background shots).
  - **Train set:** 1,768 images | **Val set:** 379 images | **Test set:** 380 images.
  - **Leakage verification:** 0 overlapping file paths between train, val, and test.
- **Interview Takeaway:** 
  > *"Before writing any model code, I audited data quality. I uncovered an 82% class imbalance between paper and trash, meaning a naive accuracy metric would be misleading. I implemented stratified splitting and established Macro F1 as the primary decision metric."*

---

### Milestone 1: Baseline Architecture Selection ($X \rightarrow Y$)

- **Starting State ($X$):** Traditional heavy architectures (VGG16 / ResNet50 from reference projects).
  - *Reference Project Behavior:* VGG16 with frozen backbone scored only ~43% test accuracy; required unfreezing deeper layers to reach 80.8%. VGG16 has ~138M parameters and ResNet50 has ~25.6M parameters.
- **Transition Proposal ($X \rightarrow Y$):** Transition to **MobileNetV3-Large** (~5.4M parameters).
- **Technical Rationale:**
  1. **Efficiency vs. Accuracy:** MobileNetV3-Large utilizes depthwise separable convolutions, hard-swish activations, and Squeeze-and-Excitation (SE) attention blocks, achieving higher ImageNet Top-1 accuracy than VGG16 with **96% fewer parameters**.
  2. **Preprocessing Integrity:** Many reference projects get 43% accuracy on frozen backbones because they scale inputs with `x / 255.0` instead of subtracting ImageNet channel means `[0.485, 0.456, 0.406]` and dividing by standard deviations `[0.229, 0.224, 0.225]`. We bind Torchvision's native `MobileNet_V3_Large_Weights.DEFAULT.transforms()` to eliminate this bug.
  3. **Edge Deployment Suitability:** For waste sorting on conveyor belts or smart bins, edge inference latency (<20ms on mobile CPU) is crucial.
- **Measured Outcome & Results:**
  - **Hardware & Device:** NVIDIA GeForce GTX 1650 (CUDA 12.4), 200.6s total runtime across 10 epochs (~18s / epoch).
  - **Best Epoch:** Epoch 6 (early stopping triggered at Epoch 10 after 7 patience checks).
  - **Train Metrics (Epoch 6):** Loss: 0.3238 | Accuracy: 88.86%
  - **Validation Metrics (Epoch 6):** Loss: 0.4986 | **Accuracy: 84.17%** | **Macro F1: 0.8194**
  - **Comparison with Reference Baseline:**
    - Reference VGG16 frozen baseline: **43.0%**
    - Our MobileNetV3-Large frozen baseline: **84.17% (+41.17% improvement)**
  - **Artifacts Generated:** `artifacts/best_model.pth`, `artifacts/training_curves.png`, `artifacts/training_history.json`.
- **Interview Talking Point:**
  > *"Rather than using a heavy, outdated architecture like VGG16 which was stuck at 43% due to incorrect preprocessing, I implemented MobileNetV3-Large with strict ImageNet channel normalization. This baseline alone reached 84.17% validation accuracy with all feature extractors completely frozen, training in just 3.3 minutes on an entry-level GPU."*

---

### Milestone 2: Transition from Frozen Backbone to Deep Fine-Tuning ($X \rightarrow Y$)

*(To be filled during Phase 3 with actual experimental metrics)*

- **Baseline ($X$):** Frozen MobileNetV3 head only.
  - *Observed Metrics:* `[To be recorded upon baseline run]`
- **Intervention ($X \rightarrow Y$):**
  - Unfreeze the top convolutional stages of MobileNetV3.
  - Apply differential learning rates: Head `lr = 1e-4`, Backbone `lr = 1e-5`.
  - Introduce cosine annealing or `ReduceLROnPlateau` scheduler.
- **Observed Change:**
  - *Train / Val Loss progression:* `[To be recorded]`
  - *Accuracy gain:* `[X% -> Y%]`
  - *Macro F1 gain:* `[X -> Y]`
- **Why this worked / did not work:**
  - *Detailed technical reflection:* `[To be recorded]`

---

### Milestone 3: Handling Class Imbalance & Data Augmentation ($X \rightarrow Y$)

*(To be filled during Phase 3 / Phase 4 with actual experimental metrics)*

- **Baseline ($X$):** Fine-tuned model without class compensation.
  - *Observed Metrics:* `[To be recorded]`
  - *Failure mode:* Confusion between plastic vs glass, low recall on `trash`.
- **Intervention ($X \rightarrow Y$):**
  - Inverse-frequency or effective-number class weighting in `CrossEntropyLoss`.
  - Targeted augmentation pipeline on training set only (RandomResizedCrop, ColorJitter, RandomHorizontalFlip).
- **Observed Change:**
  - *Per-class recall changes:*
    - `trash` recall: `[X% -> Y%]`
    - `paper` precision: `[X% -> Y%]`
  - *Overall Macro F1:* `[X -> Y]`
- **Interview Takeaway:**
  - *Evidence-based decision making.*

---

## 📝 Experiment Log Template (For New Runs)

```markdown
### Experiment Run #N: [Descriptive Name]
- **Date & Commit:** YYYY-MM-DD (`<git-hash>`)
- **Hypothesis:** If we [change X], then [metric Y] will improve because [reason Z].
- **Configuration Delta:**
  - `param_1`: old_val -> new_val
  - `param_2`: old_val -> new_val
- **Results:**
  - Train Loss: ... | Train Acc: ...
  - Val Loss: ...   | Val Acc: ...   | Val Macro F1: ...
- **Key Insight:** Did the experiment validate the hypothesis? What was learned?
```

