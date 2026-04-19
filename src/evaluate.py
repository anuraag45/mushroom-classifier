"""
evaluate.py
===========
Model evaluation with comprehensive metrics and visual reports (PyTorch).

Computes accuracy, precision, recall, F1-score per class, and generates
a confusion matrix heatmap.

Usage:
    python -m src.evaluate --model_path models/efficientnet_b2_final.pth --model_type efficientnet
    python -m src.evaluate --compare
"""

import os
import json
import argparse
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.data_preprocessing import prepare_data
from src.model import build_cnn_model, build_transfer_model, build_efficientnet_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(model_path: str, model_type: str, num_classes: int):
    """Load a trained model from checkpoint."""
    if model_type == "cnn":
        model = build_cnn_model(num_classes)
    elif model_type == "efficientnet":
        model = build_efficientnet_model(num_classes, freeze=False)
    else:
        model = build_transfer_model(num_classes, freeze=False)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


def evaluate_model(
    model_path: str,
    model_type: str = "efficientnet",
    output_dir: str = "outputs",
    data_dir: str = "data",
) -> dict:
    """Run full evaluation on the test set."""
    os.makedirs(output_dir, exist_ok=True)

    data = prepare_data(data_dir=data_dir, output_dir=output_dir)
    num_classes = data["num_classes"]
    idx_to_class = data["idx_to_class"]
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Derive model name from type
    if model_type == "cnn":
        model_name = "cnn"
    elif model_type == "efficientnet":
        model_name = "efficientnet_b2"
    else:
        model_name = "mobilenetv2"

    logger.info(f"Loading {model_name} from {model_path}...")
    model = load_trained_model(model_path, model_type, num_classes)

    # ── Predict ──
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for images, labels in data["test_loader"]:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Metrics ──
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    logger.info(f"\n{'='*60}\nClassification Report — {model_name}\n{'='*60}\n{report_str}")

    # ── Precision Warning ──
    logger.info(
        "\n⚠️  PRECISION IS CRITICAL FOR POISONOUS SPECIES\n"
        "   A false negative (predicting EDIBLE when actually POISONOUS) can be fatal.\n"
        "   High precision for poisonous classes ensures that 'EDIBLE' predictions\n"
        "   are trustworthy. Monitor per-class precision for Amanita, Cortinarius, Entoloma.\n"
    )

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {cm_path}")

    # ── Save metrics ──
    metrics = {
        "model_name": model_name,
        "model_path": model_path,
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = os.path.join(output_dir, f"{model_name}_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    return metrics


def compare_models(output_dir: str = "outputs", data_dir: str = "data") -> None:
    """Compare all available models side-by-side."""
    model_configs = [
        ("models/cnn_final.pth", "cnn"),
        ("models/mobilenetv2_final.pth", "transfer"),
        ("models/efficientnet_b2_final.pth", "efficientnet"),
    ]

    results = []
    for mp, mt in model_configs:
        if os.path.exists(mp):
            metrics = evaluate_model(mp, model_type=mt, output_dir=output_dir, data_dir=data_dir)
            results.append(metrics)

    if len(results) < 2:
        logger.warning("Need at least two models trained for comparison.")
        return

    # ── Comparison bar chart ──
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = [r["model_name"] for r in results]
    x = np.arange(len(metric_names))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, result in enumerate(results):
        values = [result[m] for m in metric_names]
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=result["model_name"], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score"])
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison chart saved to {path}")

    best_idx = np.argmax([r["f1_macro"] for r in results])
    logger.info(f"\n🏆 Best model: {results[best_idx]['model_name']} (F1={results[best_idx]['f1_macro']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate mushroom model")
    parser.add_argument("--model_path", type=str, default="models/efficientnet_b2_final.pth")
    parser.add_argument("--model_type", type=str, choices=["cnn", "transfer", "efficientnet"], default="efficientnet")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    if args.compare:
        compare_models(output_dir=args.output_dir, data_dir=args.data_dir)
    else:
        metrics = evaluate_model(args.model_path, args.model_type, args.output_dir, args.data_dir)
        print(f"\n✓ Evaluation — {metrics['model_name']}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1-Score:  {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
