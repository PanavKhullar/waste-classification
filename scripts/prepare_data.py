"""
prepare_data.py — Dataset verification, inspection, and reproducible split generation.

Features:
- Verifies or downloads TrashNet dataset (MIT License, ~41 MB).
- Performs data hygiene checks (corrupt files, unreadable formats, shape consistency).
- Computes perceptual hashes to identify exact and near-duplicate images.
- Creates stratified train (70%), validation (15%), and test (15%) splits.
- Saves split manifests (CSV) and class-to-index mappings (JSON).
"""

import os
import sys
import json
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ensure src is on sys.path for internal imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from waste_classifier.utils.reproducibility import set_seed

TRASHNET_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"
)
EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_trashnet(target_dir: Path) -> None:
    """Download and extract TrashNet dataset if not already present."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir.parent / "dataset-resized.zip"

    print(f"\n[INFO] Downloading TrashNet (~41 MB) from {TRASHNET_DOWNLOAD_URL}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="TrashNet") as t:
        urllib.request.urlretrieve(TRASHNET_DOWNLOAD_URL, filename=zip_path, reporthook=t.update_to)

    print("[INFO] Extracting dataset archive...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir.parent)

    # If extracted into dataset-resized, flatten into target_dir
    extracted_nested = target_dir.parent / "dataset-resized"
    if extracted_nested.exists() and extracted_nested != target_dir:
        for item in extracted_nested.iterdir():
            dest = target_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    import shutil
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            item.rename(dest)
        extracted_nested.rmdir()

    if zip_path.exists():
        zip_path.unlink()
    print("[INFO] Download and extraction complete.")


def calculate_dhash(image_path: Path, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) using OpenCV for duplicate detection."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def inspect_dataset(data_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[int, List[str]]]:
    """Inspect all images in data_dir, checking readability and duplicates."""
    records = []
    corrupt_files = []
    hashes = {}

    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not subdirs:
        raise ValueError(f"No class folders found in {data_dir}")

    for class_dir in sorted(subdirs):
        class_name = class_dir.name
        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ]

        print(f"[SCAN] Checking class '{class_name}' ({len(image_files)} files)...")
        for img_path in image_files:
            # 1. Check with PIL
            try:
                with Image.open(img_path) as pil_img:
                    pil_img.verify()
            except Exception:
                corrupt_files.append(str(img_path))
                continue

            # 2. Check and read dimensions with OpenCV
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                corrupt_files.append(str(img_path))
                continue

            h, w, c = cv_img.shape

            # 3. Compute perceptual hash
            img_hash = calculate_dhash(img_path)
            if img_hash is not None:
                hashes.setdefault(img_hash, []).append(str(img_path))

            # Store relative path for portability across environments/Colab
            rel_path = os.path.relpath(img_path, PROJECT_ROOT).replace("\\", "/")

            records.append({
                "filepath": rel_path,
                "filename": img_path.name,
                "class_name": class_name,
                "width": w,
                "height": h,
                "channels": c,
                "dhash": hex(img_hash) if img_hash else None,
            })

    df = pd.DataFrame(records)
    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    return df, corrupt_files, duplicates


def main():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["dataset"]["random_seed"]
    set_seed(seed, deterministic=True)

    data_raw = PROJECT_ROOT / config["paths"]["data_raw"]
    data_splits = PROJECT_ROOT / config["paths"]["data_splits"]
    data_splits.mkdir(parents=True, exist_ok=True)

    # 1. Check if dataset exists or download
    subdirs = [d for d in data_raw.iterdir() if d.is_dir() and not d.name.startswith(".")] if data_raw.exists() else []
    if not subdirs or not any(d.name in EXPECTED_CLASSES for d in subdirs):
        print(f"[INFO] TrashNet dataset not found in {data_raw}.")
        download_and_extract_trashnet(data_raw)

    # 2. Inspect dataset
    print("\n" + "=" * 60)
    print("STEP 1: DATASET INSPECTION & INTEGRITY CHECK")
    print("=" * 60)
    df, corrupt_files, duplicates = inspect_dataset(data_raw)

    print(f"\nTotal valid images found: {len(df)}")
    print(f"Corrupt / unreadable images: {len(corrupt_files)}")
    if corrupt_files:
        print("Corrupt file list:", corrupt_files)

    print(f"Exact / near-duplicate image clusters (dHash): {len(duplicates)}")
    if duplicates:
        print("Sample duplicate pairs:")
        for h, paths in list(duplicates.items())[:3]:
            print(f"  - Hash {hex(h)}: {[os.path.basename(p) for p in paths]}")

    # Class distribution analysis
    class_counts = df["class_name"].value_counts().sort_index()
    print("\nClass Distribution:")
    for cls_name, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {cls_name:<12}: {count:>4} images ({pct:>5.1f}%)")

    # Map classes to indices
    unique_classes = sorted(df["class_name"].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    idx_to_class = {idx: cls_name for idx, cls_name in enumerate(unique_classes)}
    df["class_idx"] = df["class_name"].map(class_to_idx)

    # 3. Create Stratified Splits (70% train, 15% val, 15% test)
    print("\n" + "=" * 60)
    print("STEP 2: STRATIFIED TRAIN / VAL / TEST SPLITS")
    print("=" * 60)
    ratios = config["dataset"]["split_ratios"]
    train_ratio = ratios["train"]
    val_ratio = ratios["val"]
    test_ratio = ratios["test"]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["class_name"],
        random_state=seed,
        shuffle=True,
    )

    # Second split: val vs test (equal split of the remaining 30%)
    relative_test_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=relative_test_ratio,
        stratify=val_test_df["class_name"],
        random_state=seed,
        shuffle=True,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Verify no overlap
    train_paths = set(train_df["filepath"])
    val_paths = set(val_df["filepath"])
    test_paths = set(test_df["filepath"])
    assert len(train_paths.intersection(val_paths)) == 0, "Leakage between train and val!"
    assert len(train_paths.intersection(test_paths)) == 0, "Leakage between train and test!"
    assert len(val_paths.intersection(test_paths)) == 0, "Leakage between val and test!"

    # 4. Display Split Breakdown
    split_summary = pd.crosstab(combined_df["class_name"], combined_df["split"])[["train", "val", "test"]]
    split_summary["total"] = split_summary.sum(axis=1)
    print("\nPer-Class Split Breakdown:")
    print(split_summary.to_string())

    print("\nTotal per split:")
    print(f"  Train : {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val   : {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # 5. Save manifests
    manifest_path = PROJECT_ROOT / config["paths"]["split_manifest"]
    class_mapping_path = PROJECT_ROOT / config["paths"]["class_mapping"]

    combined_df.to_csv(manifest_path, index=False)
    train_df.to_csv(data_splits / "train.csv", index=False)
    val_df.to_csv(data_splits / "val.csv", index=False)
    test_df.to_csv(data_splits / "test.csv", index=False)

    mapping_payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": len(unique_classes),
        "classes": unique_classes,
    }
    with open(class_mapping_path, "w") as f:
        json.dump(mapping_payload, f, indent=2)

    print(f"\n[SAVED] Master split manifest -> {manifest_path}")
    print(f"[SAVED] Individual split CSVs -> {data_splits}")
    print(f"[SAVED] Class mapping -> {class_mapping_path}")
    print("\n[SUCCESS] Phase 1 Data Preparation complete!")


if __name__ == "__main__":
    main()
