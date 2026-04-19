"""
utils.py
========
Utility functions: Grad-CAM explainability and confidence-based
safe prediction system.

Contains:
    1. GradCAM — visual explanation of model decisions
    2. safe_predict — confidence-thresholded prediction with safety flag
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {0: "Edible", 1: "Poisonous"}

# Same normalization used during training
EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE-BASED SAFE PREDICTION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def safe_predict(model, image, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """Run prediction with confidence-based safety check.

    CRITICAL SAFETY RULE:
        if confidence < 0.70 → prediction = "UNCERTAIN", safety = "UNSAFE"
        The system must NEVER output edible/poisonous with low confidence.

    Args:
        model: trained PyTorch model
        image: PIL Image, file path, or numpy array
        threshold: confidence threshold (default 0.70)

    Returns:
        {
            "prediction": "Edible" | "Poisonous" | "UNCERTAIN",
            "confidence": float,
            "safety": "SAFE" | "UNSAFE",
            "probabilities": {"Edible": float, "Poisonous": float},
            "warning": str | None,
        }
    """
    # Preprocess image
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    tensor = EVAL_TRANSFORMS(img).unsqueeze(0).to(DEVICE)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    # ── SAFETY DECISION ──
    if confidence < threshold:
        prediction = "UNCERTAIN"
        safety = "UNSAFE"
        warning = (
            f"⚠️ Low confidence ({confidence:.1%}). "
            "Cannot determine if edible or poisonous. "
            "Do NOT consume — consult an expert mycologist."
        )
    else:
        prediction = CLASS_NAMES[pred_idx]
        safety = "SAFE"
        warning = None

    return {
        "prediction": prediction,
        "confidence": confidence,
        "safety": safety,
        "probabilities": {
            "Edible": float(probs[0]),
            "Poisonous": float(probs[1]),
        },
        "warning": warning,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Highlights image regions that most influenced the model's prediction.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization" (ICCV 2017).

    Usage:
        cam = GradCAM(model, target_layer)
        heatmap = cam(input_tensor)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook into target layer to capture activations and gradients
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        """Forward hook: save feature map activations."""
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook: save gradients flowing back."""
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image (1, 3, H, W)
            class_idx: class to explain (default: predicted class)

        Returns:
            heatmap as (H, W) numpy array in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.to(DEVICE).requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Grad-CAM: weight activations by global-average-pooled gradients
        weights = self.gradients[0].mean(dim=(1, 2))  # (C,)
        heatmap = torch.zeros(self.activations.shape[2:], device=DEVICE)
        for i, w in enumerate(weights):
            heatmap += w * self.activations[0, i]

        # ReLU: keep only positive contributions
        heatmap = F.relu(heatmap)

        # Normalize to [0, 1]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Resize to image dimensions
        heatmap = heatmap.cpu().numpy()
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        return heatmap


def get_gradcam_overlay(model, image, alpha: float = 0.4):
    """Generate Grad-CAM heatmap overlay on original image.

    Args:
        model: trained model
        image: PIL Image
        alpha: heatmap opacity (0=original, 1=heatmap only)

    Returns:
        (overlay_image: PIL.Image, raw_heatmap: np.ndarray)
    """
    img = image.convert("RGB")
    tensor = EVAL_TRANSFORMS(img).unsqueeze(0).to(DEVICE)

    # Target layer: last feature block in MobileNetV2
    target_layer = model.backbone.features[-1]

    cam = GradCAM(model, target_layer)
    heatmap = cam(tensor)

    # Create colored overlay
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_array = np.array(img_resized)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(img_array * (1 - alpha) + heatmap_colored * alpha)
    return Image.fromarray(overlay), heatmap


if __name__ == "__main__":
    print("Utils module loaded successfully.")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Device: {DEVICE}")
