"""
explainability.py
=================
Grad-CAM (Gradient-weighted Class Activation Mapping) for PyTorch.

Generates heatmaps that highlight image regions most influential in
the model's decision, overlaid on the original image.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization" (ICCV 2017).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2

IMG_SIZE = 224
# Force CPU — no CUDA dependency (required for Streamlit Cloud deployment)
DEVICE = torch.device("cpu")

# Normalization used during inference
EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GradCAM:
    """Grad-CAM implementation for PyTorch models.

    Usage:
        cam = GradCAM(model, target_layer)
        heatmap = cam(input_tensor, class_idx=None)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor (1, 3, H, W)
            class_idx: class to explain (default: predicted class)

        Returns:
            Heatmap as (H, W) float array in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target = output[0, class_idx]
        target.backward()

        # Grad-CAM computation
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination
        heatmap = torch.zeros(activations.shape[1:], device=DEVICE)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # ReLU — only positive contributions
        heatmap = F.relu(heatmap)

        # Normalize to [0, 1]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap = heatmap.cpu().numpy()

        # Resize to image size
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

        return heatmap


def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Find the last convolutional layer in the model.

    For MushroomCNN: last Conv2d in self.features
    For MushroomMobileNet: last conv in backbone.features
    For MushroomEfficientNet: last conv in backbone.features
    """
    # Check for models with backbone (MobileNet / EfficientNet)
    if hasattr(model, 'backbone'):
        return model.backbone.features[-1]

    # Custom CNN — find last Conv2d in features
    if hasattr(model, 'features'):
        last_conv = None
        for module in model.features.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv

    # Fallback: search entire model
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv

    raise ValueError("No convolutional layer found")


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """Overlay a Grad-CAM heatmap on the original image."""
    img = original_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_array = np.array(img)

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(img_array * (1 - alpha) + colored_heatmap * alpha)
    return Image.fromarray(overlay)


def explain_prediction(
    model: torch.nn.Module,
    image_input,
    pred_index: int = None,
    alpha: float = 0.4,
) -> tuple:
    """Full explainability pipeline: generate Grad-CAM and overlay.

    Args:
        model: trained PyTorch model
        image_input: file path, PIL Image, or numpy array
        pred_index: class to explain (default: top prediction)
        alpha: heatmap overlay opacity

    Returns:
        (heatmap_overlay: PIL.Image, raw_heatmap: np.ndarray)
    """
    # Load original image
    if isinstance(image_input, str):
        original = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        original = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        original = Image.fromarray(image_input).convert("RGB")
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")

    # Preprocess
    tensor = EVAL_TRANSFORMS(original).unsqueeze(0).to(DEVICE)

    # Find target layer and create Grad-CAM
    target_layer = find_target_layer(model)
    cam = GradCAM(model, target_layer)
    heatmap = cam(tensor, class_idx=pred_index)

    # Overlay
    overlay = overlay_heatmap(original, heatmap, alpha=alpha)

    return overlay, heatmap


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanation")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/efficientnet_b2_final.pth")
    parser.add_argument("--model_type", type=str, default="efficientnet")
    parser.add_argument("--output", type=str, default="outputs/gradcam_overlay.png")
    args = parser.parse_args()

    import json
    from src.model import build_cnn_model, build_transfer_model, build_efficientnet_model

    with open("outputs/data_metadata.json") as f:
        meta = json.load(f)

    num_classes = meta["num_classes"]
    if args.model_type == "cnn":
        model = build_cnn_model(num_classes)
    elif args.model_type == "efficientnet":
        model = build_efficientnet_model(num_classes, freeze=False)
    else:
        model = build_transfer_model(num_classes, freeze=False)

    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()

    overlay, heatmap = explain_prediction(model, args.image)
    overlay.save(args.output)
    print(f"✓ Grad-CAM overlay saved to {args.output}")
