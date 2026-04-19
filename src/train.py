"""
train.py
========
Training orchestration for mushroom classification models (PyTorch).

Supports CNN, MobileNetV2, and EfficientNet-B2 models.
Includes AMP (mixed precision), label smoothing, OneCycleLR,
early stopping, and checkpoints.

Usage:
    python -m src.train --model cnn --epochs 30
    python -m src.train --model transfer --epochs 20 --fine_tune
    python -m src.train --model efficientnet --epochs 25 --fine_tune
"""

import os
import json
import argparse
import logging
import copy
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from src.data_preprocessing import prepare_data
from src.model import build_cnn_model, build_transfer_model, build_efficientnet_model, count_parameters

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if not torch.cuda.is_available():
    raise RuntimeError(
        "❌ CUDA GPU not available! This project requires an NVIDIA GPU with CUDA.\n"
        "   Install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    )
DEVICE = torch.device("cuda")
logger.info(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, scheduler=None):
    """Train for one epoch with optional AMP. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validate the model. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


class EarlyStopping:
    """Early stopping to halt training when val_loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def plot_history(history: dict, output_dir: str, model_name: str) -> None:
    """Save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_acc"], label="Train", linewidth=2)
    axes[0].plot(history["val_acc"], label="Validation", linewidth=2)
    axes[0].set_title(f"{model_name} — Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_loss"], label="Train", linewidth=2)
    axes[1].plot(history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title(f"{model_name} — Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{model_name}_training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curves saved to {path}")


def train_model(
    model_type: str = "efficientnet",
    epochs: int = 25,
    fine_tune: bool = True,
    fine_tune_epochs: int = 15,
    model_dir: str = "models",
    output_dir: str = "outputs",
    data_dir: str = "data",
    lr: float = 1e-3,
    fine_tune_lr: float = 5e-5,
) -> dict:
    """Full training pipeline with AMP, label smoothing, and OneCycleLR.

    Args:
        model_type: 'cnn', 'transfer', or 'efficientnet'
        epochs: number of initial training epochs
        fine_tune: whether to fine-tune the backbone (transfer/efficientnet only)
        fine_tune_epochs: additional epochs for fine-tuning
        model_dir: directory to save models
        output_dir: directory to save plots and metrics
        data_dir: data directory
        lr: initial learning rate
        fine_tune_lr: learning rate for fine-tuning phase

    Returns:
        Dictionary with model, history, and metadata
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── Prepare data ──
    logger.info("Preparing data...")
    data = prepare_data(data_dir=data_dir, output_dir=output_dir)
    num_classes = data["num_classes"]

    # ── Class-weighted loss with label smoothing ──
    weights = data["class_weights"]
    weight_tensor = torch.zeros(num_classes)
    for idx, w in weights.items():
        weight_tensor[idx] = w
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(DEVICE), label_smoothing=0.1)

    # ── Build model ──
    if model_type == "cnn":
        model = build_cnn_model(num_classes)
        model_name = "cnn"
    elif model_type == "efficientnet":
        model = build_efficientnet_model(num_classes)
        model_name = "efficientnet_b2"
    else:
        model = build_transfer_model(num_classes)
        model_name = "mobilenetv2"

    model = model.to(DEVICE)
    total_p, train_p = count_parameters(model)
    logger.info(f"Built {model_name} — Total: {total_p:,}  Trainable: {train_p:,}")

    # ── AMP scaler for mixed precision training ──
    scaler = GradScaler()
    logger.info("⚡ Mixed precision training (AMP) enabled")

    # ── Phase 1: Head training with OneCycleLR ──
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    steps_per_epoch = len(data["train_loader"])
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=steps_per_epoch, epochs=epochs,
        pct_start=0.1, anneal_strategy="cos"
    )
    early_stopping = EarlyStopping(patience=10)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None

    # ── Train ──
    logger.info(f"Training {model_name} for {epochs} epochs on {DEVICE}...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, data["train_loader"], criterion, optimizer, DEVICE,
            scaler=scaler, scheduler=scheduler
        )
        val_loss, val_acc = validate(model, data["val_loader"], criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        early_stopping(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(model_dir, f"{model_name}_best.pth"))

        logger.info(
            f"Epoch {epoch+1}/{epochs} — "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if early_stopping.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best weights
    if best_model_state:
        model.load_state_dict(best_model_state)

    # ── Phase 2: Fine-tune backbone (transfer/efficientnet only) ──
    if model_type in ("transfer", "efficientnet") and fine_tune:
        logger.info(f"Fine-tuning {model_name} for {fine_tune_epochs} epochs (lr={fine_tune_lr})...")

        if model_type == "efficientnet":
            model.unfreeze_from(5)  # Unfreeze last ~37% of EfficientNet
        else:
            model.unfreeze_from(12)  # Unfreeze last ~37% of MobileNetV2

        total_p, train_p = count_parameters(model)
        logger.info(f"After unfreeze — Total: {total_p:,}  Trainable: {train_p:,}")

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=fine_tune_lr, weight_decay=1e-5
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-7)
        early_stopping = EarlyStopping(patience=7)

        for epoch in range(fine_tune_epochs):
            train_loss, train_acc = train_one_epoch(
                model, data["train_loader"], criterion, optimizer, DEVICE,
                scaler=scaler, scheduler=None  # CosineAnnealing steps per epoch
            )
            val_loss, val_acc = validate(model, data["val_loader"], criterion, DEVICE)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            scheduler.step()
            early_stopping(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(model_dir, f"{model_name}_finetuned.pth"))

            logger.info(
                f"Fine-tune {epoch+1}/{fine_tune_epochs} — "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if early_stopping.should_stop:
                logger.info(f"Fine-tune early stopping at epoch {epoch+1}")
                break

        if best_model_state:
            model.load_state_dict(best_model_state)

    # ── Save final model ──
    plot_history(history, output_dir, model_name)

    final_path = os.path.join(model_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Model saved to {final_path}")

    # Save metadata
    train_meta = {
        "model_type": model_type,
        "model_name": model_name,
        "num_classes": num_classes,
        "epochs_trained": len(history["train_loss"]),
        "final_train_acc": float(history["train_acc"][-1]),
        "final_val_acc": float(history["val_acc"][-1]),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "model_path": final_path,
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = os.path.join(output_dir, f"{model_name}_train_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(train_meta, f, indent=2)

    logger.info(f"🏆 Best Val Accuracy: {best_val_acc:.4f}")

    return {
        "model": model,
        "history": history,
        "model_path": final_path,
        "metadata": train_meta,
        "data": data,
    }


def main():
    parser = argparse.ArgumentParser(description="Train mushroom classification model")
    parser.add_argument("--model", type=str, choices=["cnn", "transfer", "efficientnet"], default="efficientnet")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--fine_tune_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fine_tune_lr", type=float, default=5e-5)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    result = train_model(
        model_type=args.model,
        epochs=args.epochs,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.fine_tune_epochs,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        lr=args.lr,
        fine_tune_lr=args.fine_tune_lr,
    )

    print(f"\n✓ Training complete — {result['metadata']['model_name']}")
    print(f"  Final train acc: {result['metadata']['final_train_acc']:.4f}")
    print(f"  Final val acc:   {result['metadata']['final_val_acc']:.4f}")
    print(f"  Best val acc:    {result['metadata']['best_val_acc']:.4f}")
    print(f"  Model saved:     {result['metadata']['model_path']}")


if __name__ == "__main__":
    main()
