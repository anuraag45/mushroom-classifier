"""
eda.py
======
Exploratory Data Analysis for the mushroom image dataset.

Generates:
    - Class distribution bar chart
    - Sample image grid per class
    - Imbalance analysis with statistics

All plots saved to the /outputs directory.

Usage:
    python notebooks/eda.py
"""

import os
import sys
import random
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_preprocessing import download_dataset, find_image_root, validate_images, TOXICITY_MAP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def plot_class_distribution(labels: list, output_dir: str) -> None:
    """Bar chart of class frequencies with toxicity color-coding."""
    counts = Counter(labels)
    classes = sorted(counts.keys())
    values = [counts[c] for c in classes]
    colors = ["#2ecc71" if TOXICITY_MAP.get(c, "POISONOUS") == "EDIBLE" else "#e74c3c" for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(classes, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Mushroom Genus", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Images", fontsize=12, fontweight="bold")
    ax.set_title("Class Distribution (Green = Edible, Red = Poisonous)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "class_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Class distribution chart saved to {path}")


def plot_sample_images(paths: list, labels: list, output_dir: str, samples_per_class: int = 4) -> None:
    """Grid of sample images from each class."""
    classes = sorted(set(labels))
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(3 * samples_per_class, 3 * n_classes))
    if n_classes == 1:
        axes = [axes]

    for row_idx, cls in enumerate(classes):
        cls_paths = [p for p, l in zip(paths, labels) if l == cls]
        sampled = random.sample(cls_paths, min(samples_per_class, len(cls_paths)))
        toxicity = TOXICITY_MAP.get(cls, "POISONOUS")

        for col_idx in range(samples_per_class):
            ax = axes[row_idx][col_idx] if n_classes > 1 else axes[col_idx]
            if col_idx < len(sampled):
                try:
                    img = Image.open(sampled[col_idx]).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)

            ax.axis("off")
            if col_idx == 0:
                color = "#2ecc71" if toxicity == "EDIBLE" else "#e74c3c"
                ax.set_title(f"{cls}\n({toxicity})", fontsize=10, fontweight="bold", color=color)

    plt.suptitle("Sample Images per Class", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, "sample_images_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Sample image grid saved to {path}")


def analyze_imbalance(labels: list, output_dir: str) -> None:
    """Statistical analysis of class imbalance with imbalance ratio chart."""
    counts = Counter(labels)
    classes = sorted(counts.keys())
    values = np.array([counts[c] for c in classes])

    mean_count = np.mean(values)
    max_count = np.max(values)
    min_count = np.min(values)
    imbalance_ratio = max_count / max(min_count, 1)

    # Imbalance ratio per class (relative to mean)
    ratios = values / mean_count

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#e74c3c" if r < 0.5 or r > 2.0 else "#f39c12" if r < 0.75 or r > 1.5 else "#2ecc71" for r in ratios]
    bars = ax.barh(classes, ratios, color=colors, edgecolor="white", linewidth=0.8)

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, label="Perfect balance")
    ax.set_xlabel("Ratio to Mean Count", fontsize=12, fontweight="bold")
    ax.set_title(f"Class Imbalance Analysis (Max/Min Ratio: {imbalance_ratio:.1f}x)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "imbalance_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    summary = [
        f"\n{'='*50}",
        f"  IMBALANCE ANALYSIS",
        f"{'='*50}",
        f"  Total images:     {sum(values)}",
        f"  Number of classes: {len(classes)}",
        f"  Mean per class:    {mean_count:.0f}",
        f"  Max class:         {classes[np.argmax(values)]} ({max_count})",
        f"  Min class:         {classes[np.argmin(values)]} ({min_count})",
        f"  Imbalance ratio:   {imbalance_ratio:.1f}x",
        f"",
        f"  Strategy: Using class weights (inversely proportional",
        f"  to frequency) during training to penalize errors on",
        f"  underrepresented classes more heavily.",
        f"{'='*50}",
    ]
    for line in summary:
        logger.info(line)

    # Save summary to text file
    summary_path = os.path.join(output_dir, "eda_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary))
    logger.info(f"Imbalance analysis saved to {path}")


def plot_image_dimensions(paths: list, output_dir: str, sample_size: int = 500) -> None:
    """Scatter plot of image dimensions to check for size variation."""
    sampled = random.sample(paths, min(sample_size, len(paths)))
    widths, heights = [], []

    for p in sampled:
        try:
            img = Image.open(p)
            w, h = img.size
            widths.append(w)
            heights.append(h)
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(widths, heights, alpha=0.4, s=15, c="#3498db", edgecolors="none")
    ax.set_xlabel("Width (px)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Height (px)", fontsize=12, fontweight="bold")
    ax.set_title("Image Dimension Distribution", fontsize=14, fontweight="bold")
    ax.axvline(x=224, color="#e74c3c", linestyle="--", linewidth=1, label="Target: 224px")
    ax.axhline(y=224, color="#e74c3c", linestyle="--", linewidth=1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "image_dimensions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Dimension scatter saved to {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)

    logger.info("Starting Exploratory Data Analysis...")

    # Download / locate dataset
    raw_path = download_dataset("data")
    image_root = find_image_root(raw_path)
    logger.info(f"Image root: {image_root}")

    # Validate
    valid_paths, valid_labels, removed = validate_images(image_root)
    logger.info(f"Dataset: {len(valid_paths)} valid images, {len(removed)} removed")

    # Generate plots
    plot_class_distribution(valid_labels, OUTPUT_DIR)
    plot_sample_images(valid_paths, valid_labels, OUTPUT_DIR)
    analyze_imbalance(valid_labels, OUTPUT_DIR)
    plot_image_dimensions(valid_paths, OUTPUT_DIR)

    logger.info("\n✓ EDA complete. All plots saved to outputs/")


if __name__ == "__main__":
    main()
