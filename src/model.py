"""
model.py
========
Model architectures for mushroom image classification using PyTorch.

Provides three architectures:
    1. Baseline CNN — lightweight custom convolutional network
    2. Transfer Learning (MobileNetV2) — frozen ImageNet backbone with fine-tuning
    3. EfficientNet-B2 — compound-scaled backbone for higher accuracy
"""

import torch
import torch.nn as nn
from torchvision import models

IMG_SIZE = 224


class MushroomCNN(nn.Module):
    """Three-block custom CNN baseline.

    Architecture:
        Block 1: Conv(32, 3x3) → BN → ReLU → MaxPool → Dropout(0.25)
        Block 2: Conv(64, 3x3) → BN → ReLU → MaxPool → Dropout(0.25)
        Block 3: Conv(128, 3x3) → BN → ReLU → MaxPool → Dropout(0.25)
        Head:    AdaptiveAvgPool → Dense(256) → BN → ReLU → Dropout(0.5) → Dense(num_classes)
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MushroomMobileNet(nn.Module):
    """MobileNetV2 transfer-learning model.

    Strategy:
        - Load MobileNetV2 pre-trained on ImageNet
        - Freeze the backbone to preserve learned features
        - Replace classifier head with custom layers
        - Support selective fine-tuning of later layers

    MobileNetV2 over ResNet50:
        - ~3.4M vs ~25.6M params (7.5× smaller)
        - Comparable accuracy on small-to-medium datasets
        - Faster inference → better UX in the Streamlit app
    """

    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier — reduced dropout for better gradient flow
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_from(self, layer_idx: int = 14):
        """Unfreeze backbone layers from `layer_idx` onward for fine-tuning.

        MobileNetV2 has 19 feature blocks (0-18). Unfreezing from 14 onward
        fine-tunes the last ~25% of the backbone.
        """
        for i, block in enumerate(self.backbone.features):
            if i >= layer_idx:
                for param in block.parameters():
                    param.requires_grad = True


class MushroomEfficientNet(nn.Module):
    """EfficientNet-B2 transfer-learning model.

    Why EfficientNet-B2:
        - Compound scaling (depth + width + resolution) → better features
        - ~9.1M params — larger than MobileNetV2 but still efficient
        - Significantly higher accuracy on fine-grained classification tasks
        - Better feature representation from ImageNet pre-training
        - Input resolution: 260×260 (we use 224×224 with resize for compatibility)
    """

    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)

        # Freeze backbone features
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier head with a robust custom head
        in_features = self.backbone.classifier[1].in_features  # 1408 for B2
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_from(self, block_idx: int = 6):
        """Unfreeze EfficientNet feature blocks from `block_idx` onward.

        EfficientNet-B2 has 8 feature blocks (0-7). Unfreezing from 6 onward
        fine-tunes the last ~25% of the backbone for task-specific adaptation.
        """
        for i, block in enumerate(self.backbone.features):
            if i >= block_idx:
                for param in block.parameters():
                    param.requires_grad = True


def build_cnn_model(num_classes: int) -> MushroomCNN:
    """Build and return the custom CNN model."""
    model = MushroomCNN(num_classes)
    return model


def build_transfer_model(num_classes: int, freeze: bool = True) -> MushroomMobileNet:
    """Build and return the MobileNetV2 transfer model."""
    model = MushroomMobileNet(num_classes, freeze_backbone=freeze)
    return model


def build_efficientnet_model(num_classes: int, freeze: bool = True) -> MushroomEfficientNet:
    """Build and return the EfficientNet-B2 transfer model."""
    model = MushroomEfficientNet(num_classes, freeze_backbone=freeze)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    cnn = build_cnn_model(9)
    transfer = build_transfer_model(9)
    effnet = build_efficientnet_model(9)
    t1, tr1 = count_parameters(cnn)
    t2, tr2 = count_parameters(transfer)
    t3, tr3 = count_parameters(effnet)
    print(f"CNN          — Total: {t1:,}  Trainable: {tr1:,}")
    print(f"MobileNetV2  — Total: {t2:,}  Trainable: {tr2:,}")
    print(f"EfficientNet — Total: {t3:,}  Trainable: {tr3:,}")
