"""
data_preprocessing.py
=====================
Dataset acquisition, cleaning, splitting, and DataLoader construction
for mushroom image classification using PyTorch.

Handles:
    - Programmatic dataset download via kagglehub
    - Corrupt image detection and removal
    - Stratified train/val/test splitting (70/15/15)
    - Class-weight computation for imbalance mitigation
    - On-the-fly augmentation via torchvision transforms
"""

import os
import json
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image, ImageFile

# Allow loading of truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32

# Species-to-toxicity mapping (curated for common genera in public datasets).
# Conservative: anything uncertain defaults to POISONOUS.
TOXICITY_MAP: Dict[str, str] = {
    "Agaricus": "EDIBLE",
    "Amanita": "POISONOUS",
    "Boletus": "EDIBLE",
    "Cortinarius": "POISONOUS",
    "Entoloma": "POISONOUS",
    "Hygrocybe": "EDIBLE",
    "Lactarius": "EDIBLE",
    "Russula": "EDIBLE",
    "Suillus": "EDIBLE",
}


def download_dataset(data_dir: str = "data") -> str:
    """Download mushroom image dataset via kagglehub."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("maysee/mushrooms-classification-common-genuss-images")
        logger.info(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.warning(f"kagglehub download failed ({e}). Checking local data dir...")
        if os.path.isdir(data_dir) and any(os.scandir(data_dir)):
            return data_dir
        raise RuntimeError(
            "Could not download dataset and no local data found. "
            "Place mushroom images in data/ with one subdirectory per class."
        )


def find_image_root(dataset_path: str) -> str:
    """Walk the downloaded path to find the directory containing class subdirs."""
    for root, dirs, files in os.walk(dataset_path):
        if dirs:
            for d in dirs:
                subpath = os.path.join(root, d)
                if os.path.isdir(subpath):
                    contents = os.listdir(subpath)
                    if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) for f in contents):
                        return root
    return dataset_path


def validate_images(image_root: str) -> Tuple[List[str], List[str], List[str]]:
    """Scan all images, remove corrupted ones."""
    valid_paths, valid_labels, removed = [], [], []
    class_dirs = sorted([
        d for d in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, d))
    ])

    for class_name in class_dirs:
        class_path = os.path.join(image_root, class_name)
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                continue
            try:
                img = Image.open(fpath)
                img.verify()
                # Re-open and fully load to catch truncated files
                img = Image.open(fpath)
                img.load()
                valid_paths.append(fpath)
                valid_labels.append(class_name)
            except Exception:
                removed.append(fpath)
                logger.warning(f"Corrupt image removed: {fpath}")

    logger.info(f"Valid images: {len(valid_paths)}, Removed: {len(removed)}")
    return valid_paths, valid_labels, removed


def build_class_mappings(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str], Dict[int, str]]:
    """Create class-name ↔ index mappings and toxicity lookup."""
    unique = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(unique)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    idx_to_toxicity = {
        i: TOXICITY_MAP.get(c, "POISONOUS") for i, c in idx_to_class.items()
    }
    return class_to_idx, idx_to_class, idx_to_toxicity


def split_dataset(
    paths: List[str],
    labels: List[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Stratified train/val/test split."""
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=relative_val, stratify=train_val_labels, random_state=random_state
    )
    logger.info(f"Split — Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def compute_weights(labels: List[str], class_to_idx: Dict[str, int]) -> Dict[int, float]:
    """Compute class weights inversely proportional to frequency."""
    unique = sorted(set(labels))
    indices = np.array([class_to_idx[l] for l in labels])
    weights = compute_class_weight("balanced", classes=np.array([class_to_idx[c] for c in unique]), y=indices)
    return {class_to_idx[c]: float(w) for c, w in zip(unique, weights)}


# ── Augmentation transforms ─────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """Training augmentation pipeline (tuned for mushroom images).

    Augmentation improves generalization by exposing the model to
    transformed versions of each image:
        - RandomHorizontalFlip: different camera orientations (no vertical —
          mushrooms grow upward, flipping is unnatural)
        - RandomRotation(15°): handles slight rotations
        - RandomResizedCrop: varies scale and aspect ratio
        - ColorJitter: gentle lighting variation (low hue to preserve color cues)
        - GaussianBlur: simulates slight camera defocus
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Validation/test transforms (resize + normalize only)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Custom Dataset ───────────────────────────────────────────────────────────

class MushroomDataset(Dataset):
    """PyTorch Dataset for mushroom images."""

    def __init__(self, file_paths: List[str], labels: List[str],
                 class_to_idx: Dict[str, int], transform=None):
        self.file_paths = file_paths
        self.labels = [class_to_idx[l] for l in labels]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        try:
            img = Image.open(self.file_paths[idx]).convert("RGB")
        except Exception:
            # Return a blank image if the file is corrupt at runtime
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Main entrypoint ──────────────────────────────────────────────────────────

def prepare_data(
    data_dir: str = "data",
    output_dir: str = "outputs",
    batch_size: int = BATCH_SIZE,
) -> dict:
    """End-to-end data preparation pipeline.

    Returns a dict containing DataLoaders, mappings, class weights, and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Download / locate dataset
    raw_path = download_dataset(data_dir)
    image_root = find_image_root(raw_path)
    logger.info(f"Image root: {image_root}")

    # 2. Validate images
    valid_paths, valid_labels, removed = validate_images(image_root)
    if len(valid_paths) == 0:
        raise RuntimeError("No valid images found. Check dataset structure.")

    # 3. Build mappings
    class_to_idx, idx_to_class, idx_to_toxicity = build_class_mappings(valid_labels)
    num_classes = len(class_to_idx)
    logger.info(f"Classes ({num_classes}): {list(class_to_idx.keys())}")

    # 4. Split
    splits = split_dataset(valid_paths, valid_labels)

    # 5. Class weights
    class_weights = compute_weights(splits["train"][1], class_to_idx)

    # 6. Build DataLoaders
    train_dataset = MushroomDataset(splits["train"][0], splits["train"][1], class_to_idx, get_train_transforms())
    val_dataset = MushroomDataset(splits["val"][0], splits["val"][1], class_to_idx, get_eval_transforms())
    test_dataset = MushroomDataset(splits["test"][0], splits["test"][1], class_to_idx, get_eval_transforms())

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # 7. Save metadata
    metadata = {
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "idx_to_toxicity": {str(k): v for k, v in idx_to_toxicity.items()},
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "split_sizes": {k: len(v[0]) for k, v in splits.items()},
        "removed_images": len(removed),
        "image_root": image_root,
    }
    meta_path = os.path.join(output_dir, "data_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "idx_to_toxicity": idx_to_toxicity,
        "class_weights": class_weights,
        "num_classes": num_classes,
        "splits": splits,
        "image_root": image_root,
    }


if __name__ == "__main__":
    data = prepare_data()
    print(f"✓ Data prepared — {data['num_classes']} classes")
    for name, loader in [("train", data["train_loader"]), ("val", data["val_loader"]), ("test", data["test_loader"])]:
        print(f"  {name}: {len(loader)} batches")
