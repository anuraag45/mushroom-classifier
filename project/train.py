"""
train.py
========
Complete training pipeline for binary mushroom classification.

Execution order:
    1. Load and prepare data (binary: Edible vs Poisonous)
    2. Build MobileNetV2 model (backbone frozen)
    3. Phase 1: Train classifier head only (10 epochs)
    4. Phase 2: Unfreeze last 5 layers, fine-tune (10 epochs)
    5. Evaluate on validation set
    6. Save best model + training curves

Usage:
    python train.py
    python train.py --epochs 15 --fine_tune_epochs 10

Requires CUDA GPU.
"""

import os
import copy
import argparse
import logging
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import prepare_data, CLASS_NAMES
from model import build_model, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Device Setup ─────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "CUDA is required! Install CUDA-enabled PyTorch."
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True  # Optimizes convolution algorithms
logger.info(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")


# ── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving.

    Simple logic:
        - Track best val_loss
        - If no improvement for `patience` epochs → stop
    """

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ── Training Functions ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    """Train for one epoch with mixed precision.

    Mixed precision (AMP) speeds up training by using float16 where safe,
    while keeping float32 for numerically sensitive operations.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass with automatic mixed precision
        with autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion):
    """Validate the model. Returns loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str):
    """Save accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_acc"]) + 1)

    ax1.plot(epochs, history["train_acc"], "b-o", label="Train", markersize=3)
    ax1.plot(epochs, history["val_acc"], "r-o", label="Validation", markersize=3)
    ax1.set_title("Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Add phase separator
    phase1_end = len(history.get("phase1_epochs", []))
    if phase1_end > 0:
        ax1.axvline(x=phase1_end, color="green", linestyle="--", alpha=0.5,
                     label="Fine-tune start")

    ax2.plot(epochs, history["train_loss"], "b-o", label="Train", markersize=3)
    ax2.plot(epochs, history["val_loss"], "r-o", label="Validation", markersize=3)
    ax2.set_title("Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curves saved to {save_path}")


def plot_confusion_matrix(model, loader, save_path: str):
    """Generate and save confusion matrix."""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Edible", "Poisonous"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {save_path}")


# ── Main Training Pipeline ──────────────────────────────────────────────────

def train(
    epochs: int = 10,
    fine_tune_epochs: int = 10,
    lr: float = 1e-3,
    fine_tune_lr: float = 1e-5,
    output_dir: str = "outputs",
):
    """Full training pipeline: head training → fine-tuning → evaluation.

    Args:
        epochs: Phase 1 epochs (classifier head only)
        fine_tune_epochs: Phase 2 epochs (last 5 backbone layers + head)
        lr: Learning rate for Phase 1
        fine_tune_lr: Learning rate for Phase 2
        output_dir: Directory for saving models and plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # ════════════════════════════════════════════════════════════════════
    # STEP 1: Prepare Data
    # ════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 1: Preparing data...")
    logger.info("=" * 60)

    data = prepare_data()
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    class_weights = data["class_weights"].to(DEVICE)

    # ════════════════════════════════════════════════════════════════════
    # STEP 2: Build Model
    # ════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 2: Building model...")
    logger.info("=" * 60)

    model = build_model(freeze=True)
    model = model.to(DEVICE)
    total, trainable = count_parameters(model)
    logger.info(f"Total params: {total:,} | Trainable: {trainable:,}")

    # Loss function with class weights to handle imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Mixed precision scaler
    scaler = GradScaler("cuda")

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "phase1_epochs": [],
    }
    best_val_acc = 0.0
    best_model_state = None

    # ════════════════════════════════════════════════════════════════════
    # STEP 3: Phase 1 — Train classifier head (backbone frozen)
    # ════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info(f"PHASE 1: Training classifier head for {epochs} epochs...")
    logger.info("=" * 60)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    early_stop = EarlyStopping(patience=5)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["phase1_epochs"].append(epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "best_model.pth"))

        logger.info(
            f"[Phase 1] Epoch {epoch}/{epochs} — "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        early_stop.step(val_loss)
        if early_stop.should_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Restore best weights before fine-tuning
    if best_model_state:
        model.load_state_dict(best_model_state)

    # ════════════════════════════════════════════════════════════════════
    # STEP 4: Phase 2 — Fine-tune last 5 backbone layers
    # ════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info(f"PHASE 2: Fine-tuning for {fine_tune_epochs} epochs...")
    logger.info("=" * 60)

    model.unfreeze_last_layers(n=5)
    total, trainable = count_parameters(model)
    logger.info(f"After unfreeze — Trainable: {trainable:,}")

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=fine_tune_lr
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    early_stop = EarlyStopping(patience=5)

    for epoch in range(1, fine_tune_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "best_model.pth"))

        logger.info(
            f"[Phase 2] Epoch {epoch}/{fine_tune_epochs} — "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        early_stop.step(val_loss)
        if early_stop.should_stop:
            logger.info(f"Fine-tune early stopping at epoch {epoch}")
            break

    # Restore best weights
    if best_model_state:
        model.load_state_dict(best_model_state)

    # ════════════════════════════════════════════════════════════════════
    # STEP 5: Evaluate + Save
    # ════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 5: Final evaluation...")
    logger.info("=" * 60)

    final_loss, final_acc = validate(model, val_loader, criterion)
    logger.info(f"✓ Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"✓ Final Validation Accuracy: {final_acc:.4f}")

    # Save final model
    final_path = os.path.join(output_dir, "best_model.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Model saved to {final_path}")

    # Save plots
    plot_training_curves(history,
                         os.path.join(output_dir, "training_curves.png"))
    plot_confusion_matrix(model, val_loader,
                          os.path.join(output_dir, "confusion_matrix.png"))

    # Print summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"  Model saved to:    {final_path}")
    print(f"  Training curves:   {output_dir}/training_curves.png")
    print(f"  Confusion matrix:  {output_dir}/confusion_matrix.png")
    print("=" * 60)

    if best_val_acc >= 0.75:
        print("  ✅ TARGET MET: Accuracy ≥ 75%")
    else:
        print("  ⚠️ TARGET NOT MET: Accuracy < 75%")
    print("=" * 60)

    return model, history


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mushroom classifier")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Phase 1 epochs (head only)")
    parser.add_argument("--fine_tune_epochs", type=int, default=10,
                        help="Phase 2 epochs (fine-tuning)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Phase 1 learning rate")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5,
                        help="Phase 2 learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory for outputs")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        lr=args.lr,
        fine_tune_lr=args.fine_tune_lr,
        output_dir=args.output_dir,
    )
