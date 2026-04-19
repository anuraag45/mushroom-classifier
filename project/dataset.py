"""
dataset.py
==========
Data loading, cleaning, and DataLoader construction for binary
mushroom classification (Edible vs Poisonous).

Pipeline:
    1. Download dataset via kagglehub
    2. Validate images (remove corrupt files)
    3. Map genus labels → binary (Edible=0, Poisonous=1)
    4. Stratified train/val split (80/20)
    5. Apply augmentation transforms
    6. Build DataLoaders
"""

import os
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile

# Allow truncated images to load instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32

# Binary mapping: genus → Edible or Poisonous
# Conservative: unknown genera default to Poisonous for safety
TOXICITY_MAP = {
    "Agaricus":    0,  # Edible
    "Boletus":     0,  # Edible
    "Hygrocybe":   0,  # Edible
    "Lactarius":   0,  # Edible
    "Russula":     0,  # Edible
    "Suillus":     0,  # Edible
    "Amanita":     1,  # Poisonous
    "Cortinarius": 1,  # Poisonous
    "Entoloma":    1,  # Poisonous
}

CLASS_NAMES = {0: "Edible", 1: "Poisonous"}


# ── Dataset Download ─────────────────────────────────────────────────────────

def download_dataset(data_dir: str = "data") -> str:
    """Download mushroom dataset via kagglehub, or use local data."""
    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "maysee/mushrooms-classification-common-genuss-images"
        )
        logger.info(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.warning(f"kagglehub failed ({e}), checking local data dir...")
        if os.path.isdir(data_dir) and any(os.scandir(data_dir)):
            return data_dir
        raise RuntimeError(
            "Could not download dataset and no local data found. "
            "Place mushroom images in data/ with one subdirectory per class."
        )


def find_image_root(dataset_path: str) -> str:
    """Walk downloaded path to find the directory containing class subdirs."""
    for root, dirs, files in os.walk(dataset_path):
        if dirs:
            for d in dirs:
                subpath = os.path.join(root, d)
                if os.path.isdir(subpath):
                    contents = os.listdir(subpath)
                    if any(f.lower().endswith((".jpg", ".jpeg", ".png"))
                           for f in contents):
                        return root
    return dataset_path


# ── Image Validation ─────────────────────────────────────────────────────────

def validate_and_load_images(image_root: str) -> Tuple[List[str], List[int]]:
    """Scan all images, remove corrupted ones, return paths + binary labels."""
    valid_paths = []
    valid_labels = []
    removed_count = 0

    class_dirs = sorted([
        d for d in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, d))
    ])

    for class_name in class_dirs:
        # Map genus → binary label
        binary_label = TOXICITY_MAP.get(class_name, 1)  # default: Poisonous
        class_path = os.path.join(image_root, class_name)

        for fname in os.listdir(class_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            fpath = os.path.join(class_path, fname)
            try:
                # Two-pass validation: verify then fully load
                img = Image.open(fpath)
                img.verify()
                img = Image.open(fpath)
                img.load()
                valid_paths.append(fpath)
                valid_labels.append(binary_label)
            except Exception:
                removed_count += 1
                logger.warning(f"Corrupt image removed: {fpath}")

    logger.info(f"Valid images: {len(valid_paths)} | Removed: {removed_count}")
    edible = valid_labels.count(0)
    poisonous = valid_labels.count(1)
    logger.info(f"Class distribution — Edible: {edible}, Poisonous: {poisonous}")
    return valid_paths, valid_labels


# ── Transforms ───────────────────────────────────────────────────────────────

def get_train_transforms():
    """Training augmentation — simple but effective."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """Validation transforms — resize and normalize only."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── PyTorch Dataset ──────────────────────────────────────────────────────────

class MushroomDataset(Dataset):
    """Simple PyTorch Dataset for mushroom images."""

    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load image as RGB
        try:
            image = Image.open(self.file_paths[idx]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


# ── Main Pipeline ────────────────────────────────────────────────────────────

def prepare_data(data_dir: str = "data", batch_size: int = BATCH_SIZE) -> dict:
    """Complete data preparation pipeline.

    Returns:
        Dictionary with train_loader, val_loader, and class weights.
    """
    # 1. Download / locate dataset
    raw_path = download_dataset(data_dir)
    image_root = find_image_root(raw_path)
    logger.info(f"Image root: {image_root}")

    # 2. Validate images and get binary labels
    paths, labels = validate_and_load_images(image_root)
    if len(paths) == 0:
        raise RuntimeError("No valid images found!")

    # 3. Stratified train/val split (80/20)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.20, stratify=labels, random_state=42
    )
    logger.info(f"Split — Train: {len(train_paths)}, Val: {len(val_paths)}")

    # 4. Compute class weights for imbalanced data
    class_counts = np.bincount(train_labels, minlength=2)
    total = len(train_labels)
    class_weights = torch.tensor([total / (2 * c) for c in class_counts],
                                 dtype=torch.float32)
    logger.info(f"Class weights: Edible={class_weights[0]:.3f}, "
                f"Poisonous={class_weights[1]:.3f}")

    # 5. Build datasets
    train_dataset = MushroomDataset(train_paths, train_labels,
                                    get_train_transforms())
    val_dataset = MushroomDataset(val_paths, val_labels,
                                  get_val_transforms())

    # 6. Build DataLoaders
    # num_workers=4 on Linux/Mac, 0 on Windows to avoid multiprocessing issues
    num_workers = 0 if os.name == "nt" else 4
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "class_weights": class_weights,
        "num_train": len(train_paths),
        "num_val": len(val_paths),
    }


if __name__ == "__main__":
    data = prepare_data()
    print(f"✓ Data ready — Train: {data['num_train']}, Val: {data['num_val']}")
    print(f"  Class weights: {data['class_weights']}")
