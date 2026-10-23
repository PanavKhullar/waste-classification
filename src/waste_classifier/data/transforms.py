"""
transforms.py — Image preprocessing and augmentation pipelines.

KEY CONCEPT: Model-Specific Preprocessing
==========================================
The #1 mistake in transfer learning (and what likely caused the reference
project's initial 43% accuracy) is using wrong normalization.

Every pretrained model expects inputs normalized with the SAME statistics
used during its original ImageNet training:
  - Mean: [0.485, 0.456, 0.406]  (per-channel RGB means of ImageNet)
  - Std:  [0.229, 0.224, 0.225]  (per-channel RGB stds of ImageNet)

Using simple x/255.0 instead of these values means the pretrained features
receive inputs in a completely different distribution than they were trained
on, so they output garbage features → low accuracy.

Torchvision's new weights API handles this automatically:
  weights = MobileNet_V3_Large_Weights.DEFAULT
  preprocess = weights.transforms()  # Returns the correct Compose pipeline

RULE: Augmentation is applied ONLY to training data. Validation and test
data get resize + center crop + normalize only.
"""

from torchvision import transforms


# ImageNet normalization constants — used by all standard pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224, config: dict = None) -> transforms.Compose:
    """Build the training transform pipeline with data augmentation.

    Training transforms include random augmentations to artificially increase
    dataset diversity and reduce overfitting. Each image is randomly modified
    differently every epoch, so the model never sees the exact same image twice.

    Args:
        image_size: Target spatial size (H=W) for the model input.
        config: Optional augmentation config from config.yaml.

    Returns:
        A Compose pipeline: PIL Image → augmented Tensor (C, H, W).
    """
    aug_config = config or {}

    transform_list = [
        # RandomResizedCrop: Randomly crops a region (scale 0.8-1.0 of original area)
        # and resizes to image_size. This teaches the model to recognize objects
        # at different scales and positions, not just centered objects.
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ]

    # RandomHorizontalFlip: 50% chance to mirror the image left-right.
    # Waste items look the same flipped, so this is a safe augmentation.
    if aug_config.get("random_horizontal_flip", True):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # RandomRotation: Small rotations (±15°) simulate tilted camera angles.
    rotation = aug_config.get("random_rotation", 15)
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation))

    # ColorJitter: Randomly adjusts brightness, contrast, saturation, hue.
    # This makes the model robust to different lighting conditions —
    # crucial for waste that might be photographed indoors, outdoors, etc.
    cj = aug_config.get("color_jitter", {})
    if cj:
        transform_list.append(
            transforms.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.1),
                hue=cj.get("hue", 0.05),
            )
        )

    transform_list.extend([
        # ToTensor: Converts PIL Image (H, W, C) with values 0-255
        # to a PyTorch Tensor (C, H, W) with values 0.0-1.0.
        # This also changes the memory layout from HWC to CHW,
        # which is what PyTorch convolutions expect.
        transforms.ToTensor(),

        # Normalize: Subtracts ImageNet mean and divides by std, per channel.
        # After this, pixel values are roughly in range [-2.0, 2.0].
        # This is CRITICAL for transfer learning — the pretrained features
        # expect this exact distribution.
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transforms.Compose(transform_list)


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Build the validation/test transform pipeline (NO augmentation).

    Validation and test transforms are deterministic — every time you
    pass the same image, you get the same tensor. This ensures consistent
    evaluation metrics.

    The pipeline:
    1. Resize to 256 (slightly larger than 224)
    2. CenterCrop to 224 (take the middle square)
    3. ToTensor + Normalize

    Why resize to 256 then crop to 224 instead of directly resizing to 224?
    Direct resize distorts the aspect ratio. Resize-then-crop preserves it
    and matches how ImageNet models were evaluated.

    Args:
        image_size: Target spatial size (H=W).

    Returns:
        A Compose pipeline: PIL Image → normalized Tensor (C, H, W).
    """
    return transforms.Compose([
        transforms.Resize(image_size + 32, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
