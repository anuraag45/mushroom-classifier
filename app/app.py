"""
app.py
======
Streamlit web application for Smart Mushroom Classification
& Toxicity Risk Assessment (PyTorch backend).

Deployable on Streamlit Community Cloud with model weights
hosted on Hugging Face Hub (auto-downloaded via hf_hub_download).

Features:
    - Drag-and-drop image upload
    - Species classification with confidence score
    - Edible/Poisonous determination with color-coded badge
    - Low-confidence safety warning with POISONOUS override
    - Grad-CAM explainability heatmap overlay
    - Model comparison panel

Usage:
    streamlit run app/app.py

Testing Instructions:
    1. Delete HF cache: rm -rf ~/.cache/huggingface/hub/models--Anuraaag17--mushroom-classifier-models
    2. Run: streamlit run app/app.py
    3. Verify: spinner shows "Downloading model..." on first run
    4. Upload an image → verify prediction + confidence + safety warning
    5. Re-run: model loads instantly from cache (no download)
"""

import os
import sys
import json
import logging

import streamlit as st
import numpy as np
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.model_loader import download_model_if_needed
try:
    from src.utils.model_loader import get_hf_model_catalog
except ImportError:
    def get_hf_model_catalog() -> dict:
        return {
            "efficientnet_b2": {
                "label": "EfficientNet-B2 (Recommended)",
                "type": "efficientnet",
                "repo_id": "Anuraaag17/mushroom-classifier-models",
                "filename": "efficientnet_b2_best.pth",
            },
            "mobilenetv2": {
                "label": "MobileNetV2",
                "type": "transfer",
                "repo_id": "Anuraaag17/mushroom-classifier-models",
                "filename": "mobilenetv2_best.pth",
            },
            "cnn": {
                "label": "Custom CNN",
                "type": "cnn",
                "repo_id": "Anuraaag17/mushroom-classifier-models",
                "filename": "cnn_best.pth",
            },
        }
from src.predict import predict_single, CONFIDENCE_THRESHOLD
from src.explainability import explain_prediction
from src.model import build_cnn_model, build_transfer_model, build_efficientnet_model

logger = logging.getLogger(__name__)

# ── Force CPU (required for Streamlit Cloud — no CUDA dependency) ────────────
DEVICE = torch.device("cpu")

# ── Hugging Face Model Config ────────────────────────────────────────────────
# Embedded metadata so the app works without outputs/data_metadata.json
EMBEDDED_METADATA = {
    "num_classes": 9,
    "idx_to_class": {
        0: "Agaricus", 1: "Amanita", 2: "Boletus", 3: "Cortinarius",
        4: "Entoloma", 5: "Hygrocybe", 6: "Lactarius", 7: "Russula", 8: "Suillus"
    },
    "idx_to_toxicity": {
        0: "EDIBLE", 1: "POISONOUS", 2: "EDIBLE", 3: "POISONOUS",
        4: "POISONOUS", 5: "EDIBLE", 6: "EDIBLE", 7: "EDIBLE", 8: "EDIBLE"
    },
}


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍄 Mushroom Classifier",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2ecc71, #27ae60, #16a085);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        font-size: 1.05rem;
        color: #888;
        font-weight: 300;
    }

    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .species-name {
        font-size: 2rem;
        font-weight: 700;
        color: #ecf0f1;
        margin-bottom: 0.3rem;
    }

    .badge-edible {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        box-shadow: 0 4px 15px rgba(46,204,113,0.3);
    }

    .badge-poisonous {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        box-shadow: 0 4px 15px rgba(231,76,60,0.3);
    }

    .badge-uncertain {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        box-shadow: 0 4px 15px rgba(243,156,18,0.3);
    }

    .confidence-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #bdc3c7;
        margin-top: 1rem;
    }

    .warning-box {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: white;
        font-weight: 500;
        border-left: 5px solid #e74c3c;
        box-shadow: 0 4px 15px rgba(243,156,18,0.2);
    }

    .safety-disclaimer {
        background: rgba(231,76,60,0.1);
        border: 1px solid rgba(231,76,60,0.3);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: #e74c3c;
    }

    .metric-box {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-box h3 {
        font-size: 0.85rem;
        color: #95a5a6;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
    .metric-box p {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(46,204,113,0.4);
        border-radius: 16px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────────

def ensure_model_downloaded(repo_id: str, filename: str, label: str) -> str:
    """Download a selected model from HF Hub if not cached."""
    try:
        with st.spinner("📥 Downloading model from Hugging Face Hub (first run only)..."):
            model_path = download_model_if_needed(
                repo_id=repo_id,
                filename=filename,
            )
        return model_path
    except SystemExit:
        st.error(
            "❌ **Failed to download model weights.**\n\n"
            f"- **Model:** `{label}`\n"
            f"- **Repository:** `{repo_id}`\n"
            f"- **Filename:** `{filename}`\n\n"
            "Please check:\n"
            "1. Your internet connection\n"
            "2. The Hugging Face repo exists and is public\n"
            "3. The filename matches exactly\n\n"
            "See the console logs for details."
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {e}")
        st.stop()


@st.cache_resource
def load_model_cached(model_path: str, model_type: str, num_classes: int):
    """Load and cache the ML model. Runs on CPU only."""
    if model_type == "cnn":
        model = build_cnn_model(num_classes)
    elif model_type == "efficientnet":
        model = build_efficientnet_model(num_classes, freeze=False)
    else:
        model = build_transfer_model(num_classes, freeze=False)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model


def get_metadata():
    """Load metadata from file or use embedded defaults."""
    metadata_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "data_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
        idx_to_toxicity = {int(k): v for k, v in meta["idx_to_toxicity"].items()}
        return meta["num_classes"], idx_to_class, idx_to_toxicity
    else:
        return (
            EMBEDDED_METADATA["num_classes"],
            EMBEDDED_METADATA["idx_to_class"],
            EMBEDDED_METADATA["idx_to_toxicity"],
        )


def find_available_models() -> dict:
    """Build available model options from Hugging Face and local files."""
    available = {}
    for _, config in get_hf_model_catalog().items():
        label = f"{config['label']} [HF]"
        available[label] = {
            "type": config["type"],
            "source": "huggingface",
            "repo_id": config["repo_id"],
            "filename": config["filename"],
            "label": config["label"],
        }

    # Always offer the HF-hosted EfficientNet as the primary option
    available["🚀 EfficientNet-B2 (Recommended)"] = {
        "type": "efficientnet",
        "source": "huggingface",
    }

    # Scan for local models (optional — for dev use)
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    if os.path.isdir(models_dir):
        for fname in sorted(os.listdir(models_dir)):
            if not fname.endswith(".pth"):
                continue

            full_path = os.path.join(models_dir, fname)

            if "cnn" in fname.lower():
                label = f"🧱 Custom CNN ({fname})"
                mtype = "cnn"
            elif "efficientnet" in fname.lower():
                label = f"⚡ EfficientNet ({fname})"
                mtype = "efficientnet"
            else:
                label = f"📱 MobileNetV2 ({fname})"
                mtype = "transfer"

            available[label] = {
                "path": full_path,
                "type": mtype,
                "source": "local",
                "label": label,
            }

    available = {
        name: config
        for name, config in available.items()
        if config["source"] != "huggingface" or "filename" in config
    }
    return available


def apply_safety_layer(result: dict, threshold: float) -> dict:
    """Apply safety overrides to prediction results.

    Safety Rules:
        1. If confidence < threshold → mark as low confidence
        2. If confidence < threshold AND toxicity == EDIBLE →
           override to UNCERTAIN/POISONOUS to prevent false safety
        3. Default unknown/uncertain cases to unsafe
    """
    confidence = result["confidence"]
    is_low_confidence = confidence < threshold

    if is_low_confidence:
        result["low_confidence_warning"] = True

        if result["toxicity"] == "EDIBLE":
            # CRITICAL SAFETY: Never confidently say "edible" at low confidence
            result["safety_override"] = True
            result["original_toxicity"] = "EDIBLE"
            result["toxicity"] = "UNCERTAIN"
            result["warning_message"] = (
                f"⚠️ Low confidence ({confidence:.1%}) — prediction overridden to UNSAFE. "
                "The model predicted EDIBLE but is not confident enough. "
                "Treat this mushroom as POTENTIALLY POISONOUS. "
                "Do NOT consume without expert verification."
            )
        else:
            result["safety_override"] = False
            result["warning_message"] = (
                f"⚠️ Low confidence prediction ({confidence:.1%}). "
                "Do NOT rely on this result for safety decisions. "
                "Consult an expert mycologist before consuming any wild mushroom."
            )
    else:
        result["safety_override"] = False

    return result


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    available_models = find_available_models()
    model_choice = st.selectbox(
        "Select Model",
        list(available_models.keys()),
        index=0,
    )
    selected = available_models[model_choice]
    model_type = selected["type"]

    # Resolve model path based on source
    if selected["source"] == "huggingface":
        model_path = ensure_model_downloaded(
            repo_id=selected.get("repo_id", "Anuraaag17/mushroom-classifier-models"),
            filename=selected.get("filename", "efficientnet_b2_best.pth"),
            label=selected.get("label", model_choice),
        )
    else:
        model_path = selected["path"]
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()

    threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.70, 0.05,
                          help="Below this → safety warning")

    st.markdown("### 🔥 Grad-CAM")
    gradcam_alpha = st.slider("Heatmap Opacity", 0.1, 0.8, 0.4, 0.05)
    show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)

    st.markdown("---")

    st.caption("Running on: 🔵 CPU")

    st.markdown(
        '<div class="safety-disclaimer">'
        '🚨 <strong>Safety Disclaimer</strong><br>'
        'This tool is for educational purposes only. '
        'Never consume wild mushrooms based solely on AI predictions. '
        'Always consult a professional mycologist.'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Main Content ─────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    '<h1>🍄 Smart Mushroom Classifier</h1>'
    '<p>Computer Vision–powered Species Identification & Toxicity Risk Assessment</p>'
    '</div>',
    unsafe_allow_html=True,
)

tab_predict, tab_compare, tab_about = st.tabs(["🔍 Predict", "📊 Model Comparison", "ℹ️ About"])

with tab_predict:
    uploaded_file = st.file_uploader(
        "Upload a mushroom image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Drag and drop or click to upload.",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        num_classes, idx_to_class, idx_to_toxicity = get_metadata()
        model = load_model_cached(model_path, model_type, num_classes)

        with st.spinner("Analyzing image..."):
            result = predict_single(image, model, idx_to_class, idx_to_toxicity, threshold)

        # Apply safety layer on top of prediction
        result = apply_safety_layer(result, threshold)

        col_img, col_result = st.columns([1, 1.2])

        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if show_gradcam:
                with st.spinner("Generating Grad-CAM explanation..."):
                    try:
                        overlay, _ = explain_prediction(model, image, alpha=gradcam_alpha)
                        st.image(overlay, caption="Grad-CAM Heatmap – Model Attention", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Grad-CAM generation failed: {e}")

        with col_result:
            # Choose badge based on toxicity (including safety override)
            if result["toxicity"] == "EDIBLE":
                toxicity_badge = '<span class="badge-edible">✅ EDIBLE</span>'
            elif result["toxicity"] == "UNCERTAIN":
                toxicity_badge = '<span class="badge-uncertain">⚠️ UNCERTAIN — Treat as POISONOUS</span>'
            else:
                toxicity_badge = '<span class="badge-poisonous">☠️ POISONOUS</span>'

            st.markdown(
                f'<div class="result-card">'
                f'<div class="species-name">{result["species"]}</div>'
                f'{toxicity_badge}'
                f'</div>',
                unsafe_allow_html=True,
            )

            conf_pct = result["confidence"] * 100
            conf_color = "#2ecc71" if conf_pct >= threshold * 100 else "#e74c3c"

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(
                    f'<div class="metric-box"><h3>Confidence</h3>'
                    f'<p style="color: {conf_color}">{conf_pct:.1f}%</p></div>',
                    unsafe_allow_html=True,
                )
            with col_m2:
                st.markdown(
                    f'<div class="metric-box"><h3>Species</h3>'
                    f'<p style="color: #3498db">{result["species"]}</p></div>',
                    unsafe_allow_html=True,
                )
            with col_m3:
                tox_display = result["toxicity"]
                if tox_display == "EDIBLE":
                    tox_col = "#2ecc71"
                elif tox_display == "UNCERTAIN":
                    tox_col = "#f39c12"
                else:
                    tox_col = "#e74c3c"
                st.markdown(
                    f'<div class="metric-box"><h3>Toxicity</h3>'
                    f'<p style="color: {tox_col}">{tox_display}</p></div>',
                    unsafe_allow_html=True,
                )

            if result["low_confidence_warning"]:
                st.markdown(
                    f'<div class="warning-box">'
                    f'⚠️ <strong>LOW CONFIDENCE PREDICTION</strong><br>'
                    f'{result["warning_message"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("##### Prediction Probabilities")
            probs = result["all_probabilities"]
            for cls_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                st.progress(prob, text=f"{cls_name}: {prob:.1%}")
    else:
        st.markdown(
            '<div style="text-align: center; padding: 4rem 2rem; color: #666;">'
            '<p style="font-size: 4rem; margin-bottom: 0.5rem;">📸</p>'
            '<p style="font-size: 1.2rem; font-weight: 500;">Upload a mushroom image to get started</p>'
            '<p style="font-size: 0.9rem;">Supported formats: JPG, PNG, BMP, WebP</p>'
            '</div>',
            unsafe_allow_html=True,
        )


with tab_compare:
    st.markdown("### 📊 Model Performance")

    st.markdown("""
    | Model | Accuracy | Precision | Recall | F1-Score |
    |-------|----------|-----------|--------|----------|
    | 🧱 Custom CNN | 22.0% | 0.24 | 0.31 | 0.23 |
    | 📱 MobileNetV2 | 84.0% | 0.81 | 0.86 | 0.83 |
    | 🚀 **EfficientNet-B2** | **87.0%** | **0.83** | **0.87** | **0.85** |
    """)

    st.success("🏆 **EfficientNet-B2** achieves the best performance across all metrics.")

    # Show local eval files if available
    eval_files = {
        "cnn": os.path.join(os.path.dirname(__file__), "..", "outputs", "cnn_eval_metrics.json"),
        "mobilenetv2": os.path.join(os.path.dirname(__file__), "..", "outputs", "mobilenetv2_eval_metrics.json"),
        "efficientnet_b2": os.path.join(os.path.dirname(__file__), "..", "outputs", "efficientnet_b2_eval_metrics.json"),
    }

    loaded_metrics = {}
    for name, path in eval_files.items():
        if os.path.exists(path):
            with open(path) as f:
                loaded_metrics[name] = json.load(f)

    if loaded_metrics:
        st.markdown("#### Detailed Results (from local evaluation)")
        cols = st.columns(len(loaded_metrics))
        for i, (name, data) in enumerate(loaded_metrics.items()):
            with cols[i]:
                display_name = {"cnn": "🧱 CNN", "mobilenetv2": "📱 MobileNetV2", "efficientnet_b2": "🚀 EfficientNet"}.get(name, name)
                st.markdown(f"**{display_name}**")
                st.metric("Accuracy", f"{data['accuracy']:.4f}")
                st.metric("F1-Score", f"{data['f1_macro']:.4f}")

    # Show confusion matrices if available
    cm_files = {
        "cnn": os.path.join(os.path.dirname(__file__), "..", "outputs", "cnn_confusion_matrix.png"),
        "efficientnet_b2": os.path.join(os.path.dirname(__file__), "..", "outputs", "efficientnet_b2_confusion_matrix.png"),
    }
    available_cms = {k: v for k, v in cm_files.items() if os.path.exists(v)}
    if available_cms:
        st.markdown("#### Confusion Matrices")
        cm_cols = st.columns(len(available_cms))
        for i, (name, path) in enumerate(available_cms.items()):
            with cm_cols[i]:
                st.image(path, caption=name, use_container_width=True)

    comparison_img = os.path.join(os.path.dirname(__file__), "..", "outputs", "model_comparison.png")
    if os.path.exists(comparison_img):
        st.image(comparison_img, caption="Model Comparison Chart", use_container_width=True)


with tab_about:
    st.markdown("""
    ### 🍄 About This Project

    **Smart Mushroom Classification & Toxicity Risk Assessment** uses computer vision
    to identify mushroom species and determine edibility.

    ---

    #### 🧠 How It Works

    1. **Image Upload** → Provide a photograph
    2. **Species Classification** → Deep learning identifies the genus
    3. **Toxicity Assessment** → Genus mapped to edibility
    4. **Safety Layer** → Low-confidence edible predictions overridden to UNCERTAIN
    5. **Confidence Scoring** → Model certainty reported
    6. **Visual Explanation** → Grad-CAM shows influential regions

    ---

    #### ⚠️ Why Precision Matters

    A false negative (poisonous → edible) can be **fatal**.
    We prioritize precision for poisonous classes and issue explicit warnings below the confidence threshold.
    **Low-confidence EDIBLE predictions are automatically overridden to UNSAFE.**

    ---

    #### 🛠️ Tech Stack

    | Component | Technology |
    |-----------|-----------|
    | ML Framework | PyTorch (CPU-only for deployment) |
    | Models | CNN + MobileNetV2 + EfficientNet-B2 |
    | Explainability | Grad-CAM |
    | Web App | Streamlit |
    | Model Hosting | Hugging Face Hub (`hf_hub_download`) |
    | Deployment | Streamlit Community Cloud |

    ---

    #### 🔗 Links

    - [GitHub Repository](https://github.com/anuraag45/mushroom-classifier)
    - [Model Weights (Hugging Face)](https://huggingface.co/Anuraaag17/mushroom-classifier-models)
    """)


st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.85rem; padding: 0.5rem;">'
    '🍄 Smart Mushroom Classifier | Educational use only | '
    'Never consume wild mushrooms based on AI predictions alone'
    '</div>',
    unsafe_allow_html=True,
)
