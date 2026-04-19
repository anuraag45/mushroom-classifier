"""
app.py
======
Streamlit web application for mushroom classification.

Features:
    - Drag-and-drop image upload
    - Prediction: Edible / Poisonous / UNCERTAIN
    - Confidence score with color coding
    - Safety flag (SAFE / UNSAFE) with warning banner
    - Grad-CAM heatmap visualization
    - Safety disclaimer

Usage:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import torch
from PIL import Image

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_model
from utils import safe_predict, get_gradcam_overlay, CONFIDENCE_THRESHOLD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("outputs", "best_model.pth")


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
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
    }

    .prediction-text {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .badge-edible {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        box-shadow: 0 4px 15px rgba(46,204,113,0.3);
    }

    .badge-poisonous {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        box-shadow: 0 4px 15px rgba(231,76,60,0.3);
    }

    .badge-uncertain {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        box-shadow: 0 4px 15px rgba(243,156,18,0.3);
    }

    .metric-box {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-box h4 {
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

    .warning-box {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: white;
        font-weight: 500;
        border-left: 5px solid #e74c3c;
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

    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(46,204,113,0.4);
        border-radius: 16px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load and cache the trained model."""
    model = build_model(freeze=False)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()
    return model


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    threshold = st.slider(
        "Confidence Threshold", 0.50, 0.95, 0.70, 0.05,
        help="Below this → UNCERTAIN prediction"
    )

    st.markdown("### 🔥 Grad-CAM")
    show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
    gradcam_alpha = st.slider("Heatmap Opacity", 0.1, 0.8, 0.4, 0.05)

    st.markdown("---")
    st.markdown(
        '<div class="safety-disclaimer">'
        '🚨 <strong>Safety Disclaimer</strong><br>'
        'This tool is for <strong>educational purposes only</strong>. '
        'Never consume wild mushrooms based solely on AI predictions. '
        'Always consult a professional mycologist.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown(f"**Threshold:** `{threshold:.0%}`")


# ── Main Content ─────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    '<h1>🍄 Mushroom Classifier</h1>'
    '<p>Binary Classification with Confidence-Based Safety System</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Check model exists
if not os.path.exists(MODEL_PATH):
    st.error(
        f"❌ Trained model not found at `{MODEL_PATH}`.\n\n"
        "Run training first:\n```\npython train.py\n```"
    )
    st.stop()

model = load_model()

# ── Upload & Predict ─────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a mushroom image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Drag and drop or click to upload.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("🔍 Analyzing image..."):
        result = safe_predict(model, image, threshold=threshold)

    col_img, col_result = st.columns([1, 1.2])

    # ── Image Column ──
    with col_img:
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if show_gradcam:
            with st.spinner("Generating Grad-CAM..."):
                try:
                    overlay, _ = get_gradcam_overlay(model, image, alpha=gradcam_alpha)
                    st.image(overlay, caption="Grad-CAM — Model Attention",
                             use_container_width=True)
                except Exception as e:
                    st.warning(f"Grad-CAM failed: {e}")

    # ── Result Column ──
    with col_result:
        pred = result["prediction"]

        if pred == "Edible":
            badge = '<span class="badge-edible">✅ EDIBLE</span>'
            pred_color = "#2ecc71"
        elif pred == "Poisonous":
            badge = '<span class="badge-poisonous">☠️ POISONOUS</span>'
            pred_color = "#e74c3c"
        else:
            badge = '<span class="badge-uncertain">❓ UNCERTAIN</span>'
            pred_color = "#f39c12"

        st.markdown(
            f'<div class="result-card">'
            f'<div class="prediction-text" style="color: {pred_color}">'
            f'{pred}</div>'
            f'{badge}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Metrics row
        conf_pct = result["confidence"] * 100
        conf_color = "#2ecc71" if result["safety"] == "SAFE" else "#e74c3c"
        safety_color = "#2ecc71" if result["safety"] == "SAFE" else "#e74c3c"

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                f'<div class="metric-box"><h4>Confidence</h4>'
                f'<p style="color: {conf_color}">{conf_pct:.1f}%</p></div>',
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f'<div class="metric-box"><h4>Safety</h4>'
                f'<p style="color: {safety_color}">{result["safety"]}</p></div>',
                unsafe_allow_html=True,
            )

        # Probability bars
        st.markdown("##### Class Probabilities")
        probs = result["probabilities"]
        for cls_name, prob in sorted(probs.items(), key=lambda x: x[1],
                                     reverse=True):
            st.progress(prob, text=f"{cls_name}: {prob:.1%}")

        # Warning banner
        if result["warning"]:
            st.markdown(
                f'<div class="warning-box">{result["warning"]}</div>',
                unsafe_allow_html=True,
            )

else:
    st.markdown(
        '<div style="text-align: center; padding: 4rem 2rem; color: #666;">'
        '<p style="font-size: 4rem; margin-bottom: 0.5rem;">📸</p>'
        '<p style="font-size: 1.2rem; font-weight: 500;">'
        'Upload a mushroom image to get started</p>'
        '<p style="font-size: 0.9rem;">'
        'Supported formats: JPG, PNG, BMP, WebP</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.85rem;">'
    '🍄 Mushroom Classifier | Educational use only | '
    'Never consume wild mushrooms based on AI predictions alone'
    '</div>',
    unsafe_allow_html=True,
)
