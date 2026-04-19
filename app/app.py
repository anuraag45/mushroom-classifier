"""
app.py
======
Streamlit web application for Smart Mushroom Classification
& Toxicity Risk Assessment (PyTorch backend).

Features:
    - Drag-and-drop image upload
    - Species classification with confidence score
    - Edible/Poisonous determination with color-coded badge
    - Low-confidence safety warning
    - Grad-CAM explainability heatmap overlay
    - Model comparison panel

Usage:
    streamlit run app/app.py
"""

import os
import sys
import json

import streamlit as st
import numpy as np
import torch
import requests
from PIL import Image

MODEL_PATH = "models/efficientnet_b2_best.pth"
URL = "https://huggingface.co/Anuraag17/mushroom-classifier-models/resolve/main/efficientnet_b2_best.pth"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)

    print("Downloading model...")

    response = requests.get(URL, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download model: {response.status_code}")

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Model downloaded successfully.")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_single, CONFIDENCE_THRESHOLD
from src.explainability import explain_prediction
from src.model import build_cnn_model, build_transfer_model, build_efficientnet_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

@st.cache_resource
def load_model(model_path: str, model_type: str, num_classes: int):
    """Load and cache the ML model."""
    if model_type == "cnn":
        model = build_cnn_model(num_classes)
    elif model_type == "efficientnet":
        model = build_efficientnet_model(num_classes, freeze=False)
    else:
        model = build_transfer_model(num_classes, freeze=False)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model


@st.cache_data
def load_metadata(metadata_path: str):
    """Load and cache class metadata."""
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
    idx_to_toxicity = {int(k): v for k, v in meta["idx_to_toxicity"].items()}
    return meta["num_classes"], idx_to_class, idx_to_toxicity


def find_available_models(models_dir: str = "models") -> dict:
    """Scan for trained models."""
    available = {}
    if os.path.isdir(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith(".pth"):
                key = fname.replace("_final.pth", "").replace("_best.pth", "").replace("_finetuned.pth", "")
                if key in available:
                    continue
                if "cnn" in key.lower():
                    label = "Custom CNN"
                elif "efficientnet" in key.lower():
                    label = "EfficientNet-B2 (Transfer Learning)"
                else:
                    label = "MobileNetV2 (Transfer Learning)"
                available[label] = os.path.join(models_dir, fname)
    return available


def detect_model_type(model_path: str) -> str:
    """Auto-detect model type from file path."""
    basename = os.path.basename(model_path).lower()
    if "cnn" in basename:
        return "cnn"
    elif "efficientnet" in basename:
        return "efficientnet"
    return "transfer"


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    available_models = find_available_models()
    if available_models:
        model_choice = st.selectbox(
            "Select Model",
            list(available_models.keys()),
            index=len(available_models) - 1,
        )
        model_path = available_models[model_choice]
        model_type = detect_model_type(model_path)
    else:
        st.error("No trained models found in `models/` directory.")
        st.info("Run training first:\n```\npython -m src.train --model efficientnet --epochs 25 --fine_tune\n```")
        st.stop()

    threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.70, 0.05,
                          help="Below this → safety warning")

    st.markdown("### 🔥 Grad-CAM")
    gradcam_alpha = st.slider("Heatmap Opacity", 0.1, 0.8, 0.4, 0.05)
    show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)

    st.markdown("---")
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

        metadata_path = "outputs/data_metadata.json"
        if not os.path.exists(metadata_path):
            st.error("Metadata file not found. Please run training first.")
            st.stop()

        num_classes, idx_to_class, idx_to_toxicity = load_metadata(metadata_path)
        model = load_model(model_path, model_type, num_classes)

        with st.spinner("Analyzing image..."):
            result = predict_single(image, model, idx_to_class, idx_to_toxicity, threshold)

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
            toxicity_badge = (
                '<span class="badge-edible">✅ EDIBLE</span>'
                if result["toxicity"] == "EDIBLE"
                else '<span class="badge-poisonous">☠️ POISONOUS</span>'
            )

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
                tox_col = "#2ecc71" if result["toxicity"] == "EDIBLE" else "#e74c3c"
                st.markdown(
                    f'<div class="metric-box"><h3>Toxicity</h3>'
                    f'<p style="color: {tox_col}">{result["toxicity"]}</p></div>',
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
    st.markdown("### 📊 Model Comparison")

    # Look for all available eval metrics
    eval_files = {
        "cnn": "outputs/cnn_eval_metrics.json",
        "mobilenetv2": "outputs/mobilenetv2_eval_metrics.json",
        "efficientnet_b2": "outputs/efficientnet_b2_eval_metrics.json",
    }

    loaded_metrics = {}
    for name, path in eval_files.items():
        if os.path.exists(path):
            with open(path) as f:
                loaded_metrics[name] = json.load(f)

    if len(loaded_metrics) >= 2:
        cols = st.columns(len(loaded_metrics))
        for i, (name, data) in enumerate(loaded_metrics.items()):
            with cols[i]:
                display_name = {"cnn": "🧱 Custom CNN", "mobilenetv2": "📱 MobileNetV2", "efficientnet_b2": "🚀 EfficientNet-B2"}.get(name, name)
                st.markdown(f"#### {display_name}")
                st.metric("Accuracy", f"{data['accuracy']:.4f}")
                st.metric("Precision", f"{data['precision_macro']:.4f}")
                st.metric("Recall", f"{data['recall_macro']:.4f}")
                st.metric("F1-Score", f"{data['f1_macro']:.4f}")

        best_name = max(loaded_metrics, key=lambda k: loaded_metrics[k]["f1_macro"])
        display = {"cnn": "Custom CNN", "mobilenetv2": "MobileNetV2", "efficientnet_b2": "EfficientNet-B2"}.get(best_name, best_name)
        st.success(f"🏆 **{display}** has the best F1-Score.")

        comparison_img = "outputs/model_comparison.png"
        if os.path.exists(comparison_img):
            st.image(comparison_img, caption="Metric Comparison", use_container_width=True)

        st.markdown("#### Confusion Matrices")
        cm_cols = st.columns(len(loaded_metrics))
        cm_files = {
            "cnn": "outputs/cnn_confusion_matrix.png",
            "mobilenetv2": "outputs/mobilenetv2_confusion_matrix.png",
            "efficientnet_b2": "outputs/efficientnet_b2_confusion_matrix.png",
        }
        for i, (name, _) in enumerate(loaded_metrics.items()):
            cm_path = cm_files.get(name)
            if cm_path and os.path.exists(cm_path):
                with cm_cols[i]:
                    st.image(cm_path, caption=name, use_container_width=True)
    else:
        st.info(
            "Train and evaluate at least two models first:\n\n"
            "```bash\n"
            "python -m src.train --model efficientnet --epochs 25 --fine_tune\n"
            "python -m src.train --model transfer --epochs 20 --fine_tune\n"
            "python -m src.evaluate --compare\n"
            "```"
        )


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
    4. **Confidence Scoring** → Model certainty reported
    5. **Visual Explanation** → Grad-CAM shows influential regions

    ---

    #### ⚠️ Why Precision Matters

    A false negative (poisonous → edible) can be **fatal**.
    We prioritize precision for poisonous classes and issue explicit warnings below the confidence threshold.

    ---

    #### 🛠️ Tech Stack

    | Component | Technology |
    |-----------|-----------|
    | ML Framework | PyTorch |
    | Models | CNN + MobileNetV2 + EfficientNet-B2 |
    | Explainability | Grad-CAM |
    | Web App | Streamlit |
    """)


st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.85rem; padding: 0.5rem;">'
    '🍄 Smart Mushroom Classifier | Educational use only | '
    'Never consume wild mushrooms based on AI predictions alone'
    '</div>',
    unsafe_allow_html=True,
)
