"""
build_model.py — Model construction for waste classification.

KEY CONCEPT: Transfer Learning
================================
Instead of training a CNN from scratch on our small 2,527-image dataset
(which would severely overfit), we use a model pre-trained on ImageNet
(1.2M images, 1000 classes). The pretrained model has already learned
to extract useful visual features like edges, textures, shapes, and objects.

Transfer Learning Strategy:
  1. FROZEN BACKBONE (Phase 2): Keep all pretrained layers frozen.
     Only train a new classification head. The pretrained features are
     used as-is — like a fixed feature extractor.

  2. FINE-TUNING (Phase 3): Unfreeze some deeper layers and train them
     with a very small learning rate. This adapts the high-level features
     to waste-specific patterns (texture of crumpled paper vs. smooth plastic).

WHY MobileNetV3-Large?
  - 5.4M parameters (vs VGG16's 138M or ResNet50's 25.6M)
  - Uses depthwise separable convolutions (efficient)
  - Squeeze-and-Excitation (SE) attention blocks
  - Hard-swish activation (faster than standard swish)
  - Higher ImageNet accuracy than VGG16 with 96% fewer parameters
  - Fast inference — suitable for edge/mobile deployment
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights


def build_model(
    num_classes: int = 6,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.5,
) -> nn.Module:
    """Build a MobileNetV3-Large model with a custom classification head.

    Architecture:
      MobileNetV3-Large (pretrained on ImageNet)
        └── features: 16 inverted residual blocks (frozen or trainable)
        └── avgpool: Adaptive average pooling → 960-dim vector
        └── classifier: NEW head for our 6 waste classes
              ├── Linear(960, 1280)
              ├── Hardswish
              ├── Dropout(0.5)
              └── Linear(1280, 6)

    Args:
        num_classes: Number of output classes (6 for TrashNet).
        pretrained: Whether to load ImageNet pretrained weights.
        freeze_backbone: If True, freeze all feature extraction layers.
        dropout: Dropout probability in the classifier head.

    Returns:
        The configured model (on CPU).
    """
    # Load pretrained MobileNetV3-Large
    # The weights enum ensures we get the correct preprocessing transforms
    if pretrained:
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        print(f"[MODEL] Loaded MobileNetV3-Large with pretrained ImageNet weights")
        print(f"[MODEL] Weights version: {weights}")
    else:
        model = models.mobilenet_v3_large(weights=None)
        print(f"[MODEL] Loaded MobileNetV3-Large with random initialization")

    # STEP 1: Freeze the backbone (feature extraction layers)
    # When frozen, these layers' weights are NOT updated during training.
    # Only the new classification head learns.
    #
    # WHY? The pretrained features are already excellent for extracting
    # visual patterns. With only 1,768 training images, trying to retrain
    # 5.4M parameters would cause massive overfitting.
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        print(f"[MODEL] Backbone FROZEN — only classifier head will be trained")
    else:
        print(f"[MODEL] Backbone UNFROZEN — all layers will be trained")

    # STEP 2: Replace the classifier head
    # Original MobileNetV3 head outputs 1000 classes (ImageNet).
    # We need 6 classes (our waste categories).
    #
    # The original classifier structure is:
    #   Linear(960, 1280) → Hardswish → Dropout(0.2) → Linear(1280, 1000)
    #
    # We replace it with:
    #   Linear(960, 1280) → Hardswish → Dropout(0.5) → Linear(1280, 6)
    #
    # Higher dropout (0.5 vs 0.2) because our dataset is much smaller
    # than ImageNet, so we need stronger regularization.

    # Get the input features of the last linear layer
    in_features = model.classifier[-1].in_features  # 1280

    model.classifier[-1] = nn.Linear(in_features, num_classes)

    # Update dropout rate in the classifier
    # model.classifier[2] is the Dropout layer
    if isinstance(model.classifier[2], nn.Dropout):
        model.classifier[2] = nn.Dropout(p=dropout)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"[MODEL] Total parameters:     {total_params:>10,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:>10,}")
    print(f"[MODEL] Frozen parameters:    {frozen_params:>10,}")
    print(f"[MODEL] Output classes:       {num_classes}")

    return model


def unfreeze_backbone_layers(model: nn.Module, num_layers_to_unfreeze: int = 4) -> None:
    """Selectively unfreeze the deepest backbone layers for fine-tuning.

    MobileNetV3-Large has 17 feature blocks (indices 0-16).
    We unfreeze from the END (deepest/most abstract layers) because:
    - Early layers learn generic features (edges, textures) — useful as-is
    - Deep layers learn task-specific features — worth adapting to waste images

    Args:
        model: The MobileNetV3 model.
        num_layers_to_unfreeze: How many feature blocks to unfreeze from the end.
    """
    # First, freeze everything
    for param in model.features.parameters():
        param.requires_grad = False

    # Then unfreeze the last N blocks
    total_blocks = len(model.features)
    start_unfreeze = total_blocks - num_layers_to_unfreeze

    for i in range(start_unfreeze, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Unfroze last {num_layers_to_unfreeze} feature blocks "
          f"(blocks {start_unfreeze}-{total_blocks - 1})")
    print(f"[MODEL] Trainable parameters now: {trainable:,}")
