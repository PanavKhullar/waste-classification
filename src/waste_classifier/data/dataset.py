"""
dataset.py — Custom PyTorch Dataset for waste classification.

KEY CONCEPT: PyTorch Dataset
==============================
A Dataset is a class that tells PyTorch:
  1. How many samples exist  → __len__()
  2. How to load one sample  → __getitem__(index)

PyTorch's DataLoader then uses this to:
  - Load samples in parallel (num_workers)
  - Batch them together (batch_size)
  - Shuffle them each epoch (shuffle=True for training)
  - Move them to GPU if needed

Our WasteDataset reads from CSV manifests (train.csv, val.csv, test.csv)
instead of scanning directories at runtime. This means:
  - Splits are deterministic and reproducible
  - No accidental data leakage between splits
  - Works the same on Windows, Linux, and Colab
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class WasteDataset(Dataset):
    """PyTorch Dataset that loads waste images from a split manifest CSV.

    Each row in the CSV has columns: filepath, class_name, class_idx, etc.
    The dataset loads the image from disk, applies the given transforms,
    and returns (image_tensor, label_int).

    Args:
        csv_path: Path to split CSV (train.csv, val.csv, or test.csv).
        project_root: Root directory of the project (to resolve relative paths).
        transform: Torchvision transform pipeline to apply to each image.
    """

    def __init__(
        self,
        csv_path: str | Path,
        project_root: str | Path,
        transform: Optional[transforms.Compose] = None,
    ):
        self.project_root = Path(project_root)
        self.transform = transform

        # Load the manifest CSV
        self.df = pd.read_csv(csv_path)

        # Extract the columns we need
        self.filepaths = self.df["filepath"].tolist()
        self.labels = self.df["class_idx"].astype(int).tolist()
        self.class_names = self.df["class_name"].tolist()

    def __len__(self) -> int:
        """Return the total number of images in this split."""
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Tuple:
        """Load and return one (image_tensor, label) pair.

        This is called by DataLoader for each sample. The flow is:
        1. Read the image file path from the CSV
        2. Open the image with PIL (as RGB)
        3. Apply transforms (augment + normalize for training,
           or just normalize for val/test)
        4. Return (tensor, int_label)

        Args:
            index: Index of the sample to load.

        Returns:
            Tuple of (image_tensor [C, H, W], label_int)
        """
        # Resolve the full path from the relative path stored in CSV
        rel_path = self.filepaths[index]
        img_path = self.project_root / rel_path

        # Open image as RGB (some images might be grayscale or RGBA)
        # .convert("RGB") ensures consistent 3-channel input
        image = Image.open(img_path).convert("RGB")

        # Apply transforms (augmentation + normalization)
        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        return image, label

    def get_class_name(self, index: int) -> str:
        """Return the class name string for a given sample index."""
        return self.class_names[index]


def build_dataloaders(
    project_root: str | Path,
    splits_dir: str | Path,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict:
    """Create DataLoaders for train, val, and test splits.

    KEY CONCEPT: DataLoader
    ========================
    DataLoader wraps a Dataset and provides:
    - Batching: groups samples into batches of size `batch_size`
    - Shuffling: randomizes order each epoch (training only!)
    - Parallel loading: uses `num_workers` processes to load images
    - Pin memory: pre-allocates GPU memory for faster transfer

    Args:
        project_root: Root directory of the project.
        splits_dir: Directory containing train.csv, val.csv, test.csv.
        train_transform: Transform pipeline for training data (with augmentation).
        val_transform: Transform pipeline for val/test data (no augmentation).
        batch_size: Number of images per batch.
        num_workers: Number of parallel data loading workers.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders and dataset sizes.
    """
    project_root = Path(project_root)
    splits_dir = Path(splits_dir)

    # Create Dataset objects for each split
    train_dataset = WasteDataset(
        csv_path=splits_dir / "train.csv",
        project_root=project_root,
        transform=train_transform,
    )
    val_dataset = WasteDataset(
        csv_path=splits_dir / "val.csv",
        project_root=project_root,
        transform=val_transform,  # NO augmentation for validation
    )
    test_dataset = WasteDataset(
        csv_path=splits_dir / "test.csv",
        project_root=project_root,
        transform=val_transform,  # NO augmentation for test either
    )

    # Create DataLoaders
    # shuffle=True for training — so the model sees batches in different
    # order each epoch, preventing it from memorizing batch patterns.
    # shuffle=False for val/test — we want deterministic, comparable results.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Randomize training batch order each epoch
        num_workers=num_workers,
        pin_memory=True,    # Faster CPU→GPU transfer
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # Deterministic evaluation order
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }
