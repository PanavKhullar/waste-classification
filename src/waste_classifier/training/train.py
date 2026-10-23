"""
train.py — Main training entry point.

Ties together config, data, model, and engine into a complete training run.

Usage:
    cd waste-classification
    python -m waste_classifier.training.train

    Or from project root:
    .venv/Scripts/python -m waste_classifier.training.train
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots without display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

# Ensure src/ is on the path when run as a module
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from waste_classifier.data.transforms import get_train_transforms, get_val_transforms
from waste_classifier.data.dataset import build_dataloaders
from waste_classifier.models.build_model import build_model
from waste_classifier.training.engine import train_one_epoch, validate, EarlyStopping
from waste_classifier.utils.reproducibility import set_seed, get_device, get_environment_info
from waste_classifier.utils.metrics import compute_metrics, compute_macro_f1


def plot_training_curves(history: dict, save_path: Path) -> None:
    """Generate and save training/validation curves.

    Plots loss and accuracy curves side by side.
    These are essential for diagnosing:
    - Overfitting: train loss drops but val loss increases
    - Underfitting: both losses plateau at high values
    - Good fit: both losses decrease together and converge
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs_range, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs_range, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs_range, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs_range, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Macro F1 curve
    axes[2].plot(epochs_range, history["val_f1"], "g-o", label="Val Macro F1", markersize=4)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title("Validation Macro F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training Curves — Frozen MobileNetV3-Large Baseline", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Training curves -> {save_path}")


def main():
    # =====================================================================
    # 1. Load Configuration
    # =====================================================================
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("WASTE CLASSIFICATION — TRAINING")
    print("=" * 60)

    # =====================================================================
    # 2. Set Seeds and Device
    # =====================================================================
    seed = config["seed"]
    set_seed(seed, deterministic=config["deterministic"])

    device = get_device()
    env_info = get_environment_info()
    print(f"\n[ENV] PyTorch:    {env_info['pytorch_version']}")
    print(f"[ENV] Torchvision: {env_info['torchvision_version']}")
    print(f"[ENV] Device:      {device}")
    if env_info["cuda_device_name"]:
        print(f"[ENV] GPU:         {env_info['cuda_device_name']}")

    # =====================================================================
    # 3. Build Transforms and DataLoaders
    # =====================================================================
    image_size = config["dataset"]["image_size"]
    aug_config = config.get("augmentation", {})

    train_transform = get_train_transforms(image_size, aug_config)
    val_transform = get_val_transforms(image_size)

    print(f"\n[DATA] Image size: {image_size}x{image_size}")
    print(f"[DATA] Train transforms: {train_transform}")
    print(f"[DATA] Val transforms:   {val_transform}")

    splits_dir = PROJECT_ROOT / config["paths"]["data_splits"]
    dataloaders = build_dataloaders(
        project_root=PROJECT_ROOT,
        splits_dir=splits_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    print(f"[DATA] Train: {dataloaders['train_size']} images, "
          f"{len(dataloaders['train'])} batches")
    print(f"[DATA] Val:   {dataloaders['val_size']} images, "
          f"{len(dataloaders['val'])} batches")

    # =====================================================================
    # 4. Build Model
    # =====================================================================
    print()
    model = build_model(
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
        dropout=config["model"]["dropout"],
    )
    model = model.to(device)

    # =====================================================================
    # 5. Loss Function, Optimizer, Scheduler
    # =====================================================================
    # CrossEntropyLoss: The standard loss for multi-class classification.
    # It expects RAW LOGITS (not softmax outputs) and integer labels.
    # Internally it applies log-softmax + negative log-likelihood.
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: Adaptive learning rate optimizer.
    # We only pass parameters that require gradients (the unfrozen ones).
    # filter(lambda p: p.requires_grad, ...) skips frozen backbone params.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.
    # If val_loss hasn't improved for `patience` epochs, multiply LR by `factor`.
    # This helps escape shallow local minima by taking smaller steps.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",           # We're monitoring loss (lower is better)
        factor=config["training"]["lr_scheduler_factor"],
        patience=config["training"]["lr_scheduler_patience"],
    )

    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
    )

    print(f"\n[TRAIN] Optimizer: Adam (lr={config['training']['learning_rate']}, "
          f"weight_decay={config['training']['weight_decay']})")
    print(f"[TRAIN] Scheduler: ReduceLROnPlateau (factor={config['training']['lr_scheduler_factor']}, "
          f"patience={config['training']['lr_scheduler_patience']})")
    print(f"[TRAIN] Early stopping patience: {config['training']['early_stopping_patience']}")

    # =====================================================================
    # 6. Training Loop
    # =====================================================================
    num_epochs = config["training"]["epochs"]
    artifacts_dir = PROJECT_ROOT / config["paths"]["artifacts"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }

    best_val_f1 = 0.0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Starting training for up to {num_epochs} epochs...")
    print(f"{'='*60}\n")

    total_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{num_epochs} (lr={current_lr:.6f})")

        # --- Train ---
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )

        # --- Validate ---
        val_loss, val_acc, val_true, val_pred = validate(
            model, dataloaders["val"], criterion, device
        )
        val_f1 = compute_macro_f1(val_true, val_pred)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        epoch_time = time.time() - epoch_start

        # Track best model by validation Macro F1
        improved = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            improved = " ** BEST **"

            # Save best checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "config": config,
            }
            torch.save(checkpoint, artifacts_dir / "best_model.pth")

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f}{improved}")
        print(f"  Time: {epoch_time:.1f}s")

        # Step the scheduler (based on validation loss)
        scheduler.step(val_loss)

        # Check early stopping (based on validation loss)
        early_stopping(val_loss, model)
        if early_stopping.should_stop:
            print(f"\n[EARLY STOP] No improvement for {early_stopping.patience} epochs. "
                  f"Stopping at epoch {epoch}.")
            break

        print()

    total_time = time.time() - total_start

    # =====================================================================
    # 7. Training Summary
    # =====================================================================
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best epoch: {best_epoch}")
    print(f"Best Val Accuracy: {history['val_acc'][best_epoch - 1]:.4f}")
    print(f"Best Val Macro F1: {best_val_f1:.4f}")
    print(f"Best checkpoint:   {artifacts_dir / 'best_model.pth'}")

    # =====================================================================
    # 8. Save Training Curves and History
    # =====================================================================
    plot_training_curves(history, artifacts_dir / "training_curves.png")

    # Save raw history as JSON for later analysis
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    history_serializable["best_epoch"] = best_epoch
    history_serializable["best_val_f1"] = float(best_val_f1)
    history_serializable["total_time_seconds"] = total_time
    history_serializable["config"] = config

    with open(artifacts_dir / "training_history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)
    print(f"[SAVED] Training history -> {artifacts_dir / 'training_history.json'}")

    # Restore best weights
    early_stopping.load_best_weights(model)
    print("\n[INFO] Best model weights restored.")
    print("[NEXT] Run Phase 4 evaluation on the test set.")


if __name__ == "__main__":
    main()
