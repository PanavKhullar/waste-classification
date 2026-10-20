"""
transforms.py — Image transforms and augmentation pipelines.

Responsibilities:
- Define model-specific preprocessing (using pretrained weights API).
- Training augmentations (flip, rotation, color jitter, etc.).
- Validation/test transforms (resize + normalize only, NO augmentation).

Key rule: Never augment validation or test data.

Will be implemented in Phase 2.
"""

