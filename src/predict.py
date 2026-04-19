"""
predict.py
==========
Inference module for single-image and batch prediction with
risk-based output and confidence thresholding (PyTorch).

Usage:
    python -m src.predict --image path/to/mushroom.jpg
    python -m src.predict --batch_dir path/to/images/
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Union

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from src.model import build_cnn_model, build_transfer_model, build_efficientnet_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
# Force CPU — no CUDA dependency (required for Streamlit Cloud deployment)
DEVICE = torch.device("cpu")

# Standard ImageNet normalization
EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model_and_metadata(
    model_path: str = "models/efficientnet_b2_final.pth",
    metadata_path: str = "outputs/data_metadata.json",
    model_type: str = None,
) -> tuple:
    """Load the trained model and class metadata."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    num_classes = metadata["num_classes"]
    idx_to_class = {int(k): v for k, v in metadata["idx_to_class"].items()}
    idx_to_toxicity = {int(k): v for k, v in metadata["idx_to_toxicity"].items()}

    # Auto-detect model type from filename
    if model_type is None:
        basename = os.path.basename(model_path).lower()
        if "cnn" in basename:
            model_type = "cnn"
        elif "efficientnet" in basename:
            model_type = "efficientnet"
        else:
            model_type = "transfer"

    if model_type == "cnn":
        model = build_cnn_model(num_classes)
    elif model_type == "efficientnet":
        model = build_efficientnet_model(num_classes, freeze=False)
    else:
        model = build_transfer_model(num_classes, freeze=False)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    return model, idx_to_class, idx_to_toxicity


def preprocess_image(image_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
    """Preprocess a single image for model input.

    Accepts a file path, PIL Image, or numpy array.
    Returns a (1, 3, 224, 224) tensor on DEVICE.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert("RGB")
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")

    tensor = EVAL_TRANSFORMS(img).unsqueeze(0).to(DEVICE)
    return tensor


def predict_single(
    image_input: Union[str, Image.Image, np.ndarray],
    model: torch.nn.Module,
    idx_to_class: Dict[int, str],
    idx_to_toxicity: Dict[int, str],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> Dict:
    """Run prediction on a single image.

    Returns:
        {
            "species": str,
            "toxicity": "EDIBLE" | "POISONOUS",
            "confidence": float,
            "all_probabilities": {class_name: float, ...},
            "low_confidence_warning": bool,
            "warning_message": str | None,
        }
    """
    tensor = preprocess_image(image_input)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    species = idx_to_class[pred_idx]
    toxicity = idx_to_toxicity[pred_idx]

    low_confidence = confidence < threshold
    warning_msg = None

    # ── Safety Layer ──────────────────────────────────────────────────────
    # CRITICAL: Prevent confident "EDIBLE" predictions at low confidence.
    # Default unknown/uncertain cases to POISONOUS for safety.
    if low_confidence:
        if toxicity == "EDIBLE":
            # Override edible to poisonous when confidence is low
            toxicity = "POISONOUS"
            warning_msg = (
                f"⚠️ Low confidence ({confidence:.1%}) — safety override applied. "
                "Model predicted EDIBLE but is not confident enough. "
                "Defaulting to POISONOUS. "
                "Do NOT consume without expert verification."
            )
        else:
            warning_msg = (
                f"⚠️ Low confidence prediction ({confidence:.1%}). "
                "Do NOT rely on this result for safety decisions. "
                "Consult an expert mycologist before consuming any wild mushroom."
            )

    all_probs = {idx_to_class[i]: float(p) for i, p in enumerate(probs)}

    return {
        "species": species,
        "toxicity": toxicity,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "low_confidence_warning": low_confidence,
        "warning_message": warning_msg,
    }


def predict_batch(
    image_dir: str,
    model: torch.nn.Module,
    idx_to_class: Dict[int, str],
    idx_to_toxicity: Dict[int, str],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> List[Dict]:
    """Run prediction on all images in a directory."""
    results = []
    supported = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(supported):
            continue
        fpath = os.path.join(image_dir, fname)
        try:
            result = predict_single(fpath, model, idx_to_class, idx_to_toxicity, threshold)
            result["filename"] = fname
            results.append(result)
            status = "⚠️" if result["low_confidence_warning"] else "✓"
            logger.info(f"{status} {fname}: {result['species']} ({result['toxicity']}) — {result['confidence']:.1%}")
        except Exception as e:
            logger.error(f"Failed to predict {fname}: {e}")
            results.append({"filename": fname, "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(description="Mushroom image prediction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--batch_dir", type=str, help="Directory for batch prediction")
    parser.add_argument("--model_path", type=str, default="models/efficientnet_b2_final.pth")
    parser.add_argument("--metadata_path", type=str, default="outputs/data_metadata.json")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    model, idx_to_class, idx_to_toxicity = load_model_and_metadata(args.model_path, args.metadata_path)

    if args.image:
        result = predict_single(args.image, model, idx_to_class, idx_to_toxicity, args.threshold)
        print(f"\n{'='*50}")
        print(f"  Species:    {result['species']}")
        print(f"  Toxicity:   {result['toxicity']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        if result["warning_message"]:
            print(f"\n  {result['warning_message']}")
        print(f"{'='*50}")
    else:
        results = predict_batch(args.batch_dir, model, idx_to_class, idx_to_toxicity, args.threshold)
        print(f"\n✓ Batch prediction complete — {len(results)} images")
        out_path = os.path.join("outputs", "batch_predictions.json")
        os.makedirs("outputs", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
