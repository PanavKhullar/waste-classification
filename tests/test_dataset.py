"""
test_dataset.py — Unit tests for data loading, model architecture, and checkpoint integrity.

Run with:
    pytest tests/test_dataset.py
"""

import json
from pathlib import Path
import pytest
import torch
import pandas as pd

from waste_classifier.data.dataset import WasteDataset, build_dataloaders
from waste_classifier.data.transforms import get_train_transforms, get_val_transforms
from waste_classifier.models.build_model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_class_mapping_integrity():
    """Verify class_mapping.json has 6 classes and symmetric bidirectional mapping."""
    mapping_path = PROJECT_ROOT / "data" / "splits" / "class_mapping.json"
    assert mapping_path.exists(), "class_mapping.json is missing!"

    with open(mapping_path) as f:
        data = json.load(f)

    assert data["num_classes"] == 6
    classes = data["classes"]
    assert len(classes) == 6

    # Verify bidirectional consistency
    class_to_idx = data["class_to_idx"]
    idx_to_class = data["idx_to_class"]
    for cls in classes:
        idx = class_to_idx[cls]
        assert idx_to_class[str(idx)] == cls


def test_split_manifests_exist_and_no_leakage():
    """Verify train, val, and test manifests exist and have zero overlap."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    assert len(train_df) == 1768
    assert len(val_df) == 379
    assert len(test_df) == 380

    train_files = set(train_df["filepath"])
    val_files = set(val_df["filepath"])
    test_files = set(test_df["filepath"])

    assert len(train_files.intersection(val_files)) == 0, "Data leakage between train and val!"
    assert len(train_files.intersection(test_files)) == 0, "Data leakage between train and test!"
    assert len(val_files.intersection(test_files)) == 0, "Data leakage between val and test!"


def test_dataset_item_shape_and_dtype():
    """Verify WasteDataset returns (3, 224, 224) float32 tensor and valid int label."""
    transform = get_val_transforms(image_size=224)
    dataset = WasteDataset(
        csv_path=PROJECT_ROOT / "data" / "splits" / "train.csv",
        project_root=PROJECT_ROOT,
        transform=transform,
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224), f"Expected (3, 224, 224) but got {image.shape}"
    assert image.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label < 6


def test_dataloader_batch_dimensions():
    """Verify DataLoader yields properly batched tensors."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    loaders = build_dataloaders(
        project_root=PROJECT_ROOT,
        splits_dir=splits_dir,
        train_transform=get_train_transforms(224),
        val_transform=get_val_transforms(224),
        batch_size=8,
        num_workers=0,
    )

    images, labels = next(iter(loaders["train"]))
    assert images.shape == (8, 3, 224, 224)
    assert labels.shape == (8,)


def test_model_forward_pass_output_shape():
    """Verify MobileNetV3 model forward pass outputs (batch_size, 6) logits."""
    model = build_model(num_classes=6, pretrained=False, freeze_backbone=True)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (4, 6), f"Expected output shape (4, 6), got {logits.shape}"


def test_saved_checkpoint_reloading():
    """Verify the saved best_model.pth checkpoint loads cleanly into the architecture."""
    ckpt_path = PROJECT_ROOT / "artifacts" / "best_model.pth"
    if not ckpt_path.exists():
        pytest.skip("artifacts/best_model.pth not yet created")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "val_acc" in checkpoint
    assert "val_f1" in checkpoint

    model = build_model(num_classes=6, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Test inference on reloaded model
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 6)
