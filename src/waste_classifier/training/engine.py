"""
engine.py — Training and validation loop logic.

This file contains the core training machinery. Every important PyTorch
concept is explained inline where it first appears.

KEY CONCEPTS COVERED:
  - model.train() vs model.eval()
  - Forward pass and loss computation
  - loss.backward() and autograd
  - optimizer.step() and optimizer.zero_grad()
  - torch.no_grad() / inference_mode for evaluation
  - Gradient accumulation and batch processing
  - Early stopping
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one full training epoch.

    One epoch = the model sees every training image exactly once.
    With batch_size=32 and 1,768 training images, that's ~56 batches per epoch.

    Args:
        model: The neural network.
        dataloader: Training DataLoader (shuffled, with augmentation).
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimizer (Adam) that updates weights.
        device: CPU or CUDA device.

    Returns:
        Tuple of (average_loss, accuracy) for this epoch.
    """
    # ========================================================================
    # model.train() — Set the model to TRAINING mode
    # ========================================================================
    # This activates:
    #   - Dropout: randomly zeroes some neurons (regularization)
    #   - BatchNorm: uses batch statistics (mean/variance of current batch)
    # If you forget this, Dropout is disabled and BatchNorm uses running
    # stats from previous epochs → inconsistent training behavior.
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="  Training", leave=False)

    for batch_idx, (images, labels) in enumerate(progress_bar):
        # Move data to the same device as the model (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)

        # ====================================================================
        # STEP 1: optimizer.zero_grad() — Reset gradients to zero
        # ====================================================================
        # PyTorch ACCUMULATES gradients by default. If you don't zero them,
        # the gradients from this batch ADD to gradients from previous batches.
        # This is occasionally useful (gradient accumulation for large batches)
        # but usually you want fresh gradients for each batch.
        optimizer.zero_grad()

        # ====================================================================
        # STEP 2: Forward pass — compute predictions
        # ====================================================================
        # images shape: (batch_size, 3, 224, 224)
        # outputs shape: (batch_size, num_classes) — raw logits (unnormalized scores)
        #
        # The model passes images through:
        #   Conv layers → BatchNorm → Activations → Pooling → Classifier head
        # Each layer's computation is tracked by autograd for backpropagation.
        outputs = model(images)

        # ====================================================================
        # STEP 3: Compute loss
        # ====================================================================
        # CrossEntropyLoss internally applies:
        #   1. LogSoftmax (converts logits to log-probabilities)
        #   2. NLLLoss (negative log-likelihood)
        #
        # DO NOT apply softmax before CrossEntropyLoss — it applies it internally.
        # Applying softmax twice gives numerically unstable, wrong gradients.
        #
        # outputs: (batch_size, 6) — raw logits
        # labels:  (batch_size,)   — integer class indices (0-5)
        loss = criterion(outputs, labels)

        # ====================================================================
        # STEP 4: loss.backward() — Backpropagation
        # ====================================================================
        # This is where the magic happens. PyTorch's autograd engine:
        #   1. Traces backward through every operation that created `loss`
        #   2. Computes the gradient of loss w.r.t. every trainable parameter
        #   3. Stores these gradients in param.grad for each parameter
        #
        # After this call, every trainable weight has a .grad attribute
        # telling it "how much the loss changes if I change this weight slightly."
        loss.backward()

        # ====================================================================
        # STEP 5: optimizer.step() — Update weights
        # ====================================================================
        # The optimizer reads the gradients (param.grad) and updates the weights
        # using the chosen algorithm. For Adam:
        #   - Maintains per-parameter learning rates (adaptive)
        #   - Uses momentum (exponential moving average of past gradients)
        #   - Handles sparse gradients well
        #
        # Formula (simplified): weight_new = weight_old - lr * gradient
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)

        # Get predictions: the class with highest logit score
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        batch_loss = loss.item()
        batch_acc = (predicted == labels).float().mean().item()
        progress_bar.set_postfix({"loss": f"{batch_loss:.4f}", "acc": f"{batch_acc:.3f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Run validation/evaluation (no gradient computation).

    ========================================================================
    @torch.no_grad() — Disable gradient tracking
    ========================================================================
    During validation, we don't need gradients because we're not updating
    weights. Disabling them:
      - Saves memory (no gradient tensors stored)
      - Runs faster (no autograd computation graph)

    You can also use `with torch.inference_mode():` which is even faster
    but less flexible. @torch.no_grad() is the standard approach.

    Args:
        model: The neural network.
        dataloader: Validation or test DataLoader (not shuffled, no augmentation).
        criterion: Loss function.
        device: CPU or CUDA device.

    Returns:
        Tuple of (avg_loss, accuracy, all_true_labels, all_predicted_labels).
    """
    # ========================================================================
    # model.eval() — Set the model to EVALUATION mode
    # ========================================================================
    # This deactivates:
    #   - Dropout: all neurons are active (no random zeroing)
    #   - BatchNorm: uses the running mean/variance computed during training
    #     (not the current batch stats)
    #
    # If you forget model.eval(), Dropout randomly drops neurons during
    # validation → noisy, unreliable metrics. BatchNorm uses batch stats
    # instead of learned stats → different results with different batch sizes.
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="  Validating", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass only — no backward pass needed
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect predictions for computing F1, confusion matrix later
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_labels, all_preds


class EarlyStopping:
    """Stop training when validation loss stops improving.

    WHY EARLY STOPPING?
    Without it, the model will keep training even after it starts overfitting
    (training loss decreases but validation loss increases). Early stopping
    watches the validation loss and stops when it hasn't improved for
    `patience` consecutive epochs.

    It also saves the best model weights, so even if we train a few epochs
    past the optimum, we can restore the best checkpoint.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum change in val_loss to count as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """Check if training should stop.

        Args:
            val_loss: Current epoch's validation loss.
            model: The model (to save best weights).
        """
        if self.best_loss is None:
            # First epoch — save as baseline
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            # Improvement! Reset counter and save new best
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def load_best_weights(self, model: nn.Module) -> None:
        """Restore the best model weights after training."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
