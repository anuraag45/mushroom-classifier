"""
model.py
========
MobileNetV2 transfer learning model for binary mushroom classification.

Architecture:
    - Backbone: MobileNetV2 pretrained on ImageNet (3.4M params)
    - Head: Dropout → Linear(1280, 128) → ReLU → Dropout → Linear(128, 2)

Why MobileNetV2:
    - Lightweight (~3.4M params vs ResNet50's ~25M)
    - Strong transfer learning performance on small datasets
    - Fast inference → good UX in Streamlit app
"""

import torch.nn as nn
from torchvision import models


class MushroomClassifier(nn.Module):
    """MobileNetV2-based binary classifier for mushrooms.

    Args:
        freeze_backbone: If True, freeze all backbone layers initially.
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # Step 1: Freeze backbone to preserve ImageNet features
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Step 2: Replace classifier head for 2 classes
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_last_layers(self, n: int = 5):
        """Unfreeze the last `n` feature blocks for fine-tuning.

        MobileNetV2 has 19 feature blocks (indices 0–18).
        Unfreezing last 5 = blocks 14–18 (~25% of backbone).
        """
        total_blocks = len(self.backbone.features)
        start_idx = total_blocks - n
        for i, block in enumerate(self.backbone.features):
            if i >= start_idx:
                for param in block.parameters():
                    param.requires_grad = True


def build_model(freeze: bool = True) -> MushroomClassifier:
    """Build and return the mushroom classifier."""
    return MushroomClassifier(freeze_backbone=freeze)


def count_parameters(model: nn.Module) -> tuple:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = build_model()
    total, trainable = count_parameters(model)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {total - trainable:,}")
