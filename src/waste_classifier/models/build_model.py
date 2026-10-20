"""
build_model.py — Model construction for waste classification.

Responsibilities:
- Load pretrained MobileNetV3-Large from torchvision.
- Replace the final classifier head for our number of classes.
- Support freezing/unfreezing backbone layers.
- Use correct input preprocessing tied to pretrained weights.

Will be implemented in Phase 2.
"""

