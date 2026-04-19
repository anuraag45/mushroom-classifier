# 🍄 Smart Mushroom Classification & Toxicity Risk Assessment

A production-grade computer vision system that classifies mushroom images into species and predicts whether they are **edible** or **poisonous**, complete with Grad-CAM visual explainability and a Streamlit web interface.

**Model weights are hosted on [Hugging Face Hub](https://huggingface.co/Anuraaag17/mushroom-classifier-models)** — the app downloads them automatically on first run.

![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?logo=streamlit)
![Hugging Face](https://img.shields.io/badge/🤗_Model-Hugging_Face-yellow)

---

## 🎯 Problem Statement

Misidentifying poisonous mushrooms kills hundreds of people worldwide every year. This project builds an AI-powered safety tool that:

1. **Classifies** mushroom images into species (multi-class classification)
2. **Predicts** edibility (binary: EDIBLE vs POISONOUS)
3. **Warns** users when prediction confidence is low
4. **Safety Override** — low-confidence EDIBLE predictions are automatically overridden to UNSAFE
5. **Explains** decisions visually using Grad-CAM heatmaps

> ⚠️ **Disclaimer**: This tool is for educational purposes only. Never consume wild mushrooms based solely on AI predictions.

---

## 🛠️ Tech Stack

| Component        | Technology                                              |
|------------------|---------------------------------------------------------|
| ML Framework     | PyTorch / torchvision (CPU-only for deployment)         |
| Models           | Custom CNN · MobileNetV2 · **EfficientNet-B2** (recommended) |
| Model Hosting    | [Hugging Face Hub](https://huggingface.co/Anuraaag17/mushroom-classifier-models) via `hf_hub_download` |
| Explainability   | Grad-CAM (hook-based)                                   |
| Web App          | Streamlit                                               |
| Deployment       | Streamlit Community Cloud                               |
| Training         | AMP (Mixed Precision) · AdamW · OneCycleLR · Label Smoothing |

---

## 📁 Project Structure

```
mushroom-classifier/
├── app/
│   └── app.py                  # Streamlit web application
├── src/
│   ├── __init__.py
│   ├── model.py                # CNN, MobileNetV2, and EfficientNet-B2 architectures
│   ├── predict.py              # Inference with safety layer
│   ├── explainability.py       # Grad-CAM heatmap generation
│   ├── train.py                # Training loop (dev only)
│   ├── evaluate.py             # Metrics & comparison (dev only)
│   ├── data_preprocessing.py   # Download, cleaning, splits (dev only)
│   └── utils/
│       ├── __init__.py
│       └── model_loader.py     # HF Hub model download utility
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── requirements.txt            # Deployment dependencies
├── .gitignore
├── LICENSE
└── README.md
```

> **Note:** `data/`, `models/`, `outputs/` directories are gitignored. Model weights live on Hugging Face.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/anuraag45/mushroom-classifier.git
cd mushroom-classifier
pip install -r requirements.txt
```

### 2. Launch the App

```bash
streamlit run app/app.py
```

On first run, model weights (~35 MB) are **automatically downloaded** from Hugging Face Hub. Subsequent runs use the cached copy.

---

## 🤗 Hugging Face Model

Model weights are hosted at: **[Anuraaag17/mushroom-classifier-models](https://huggingface.co/Anuraaag17/mushroom-classifier-models)**

| File | Model | Accuracy |
|------|-------|----------|
| `efficientnet_b2_best.pth` | EfficientNet-B2 | **87.0%** |

The `model_loader.py` utility handles downloads using `hf_hub_download`:

```python
from src.utils.model_loader import download_model_if_needed

# Downloads from HF Hub if not cached, returns local path
model_path = download_model_if_needed(
    repo_id="Anuraaag17/mushroom-classifier-models",
    filename="efficientnet_b2_best.pth",
)
```

**No raw HTTP or urllib** — uses the official Hugging Face SDK with automatic caching, ETag validation, and resumable downloads.

---

## ☁️ Streamlit Cloud Deployment

### Deploy in 3 Steps

1. **Fork** this repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create new app:
   - **Repository:** `your-username/mushroom-classifier`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
4. Click **Deploy** 🚀

### Why It Works

- ✅ **No `.pth` files in repo** — model downloads from HF Hub at runtime
- ✅ **CPU-only** — no CUDA/GPU dependency
- ✅ **No hardcoded paths** — HF Hub manages caching
- ✅ **No local file dependency** — embedded metadata as fallback
- ✅ **`requirements.txt` complete** — all deps declared

---

## 🔒 Safety Layer

The system implements a strict safety-first approach:

| Scenario | Confidence | Model Says | App Shows |
|----------|-----------|------------|-----------|
| High confidence | ≥ 70% | EDIBLE | ✅ EDIBLE |
| High confidence | ≥ 70% | POISONOUS | ☠️ POISONOUS |
| Low confidence | < 70% | POISONOUS | ☠️ POISONOUS + ⚠️ warning |
| Low confidence | < 70% | EDIBLE | **⚠️ UNCERTAIN → Treat as POISONOUS** |

> **Critical Rule:** A low-confidence EDIBLE prediction is **never** shown as safe. It is automatically overridden to UNCERTAIN/POISONOUS with a warning.

---

## 📊 Model Architectures

### Baseline: Custom CNN

```
Conv(32) → BN → ReLU → MaxPool → Dropout(0.25)
Conv(64) → BN → ReLU → MaxPool → Dropout(0.25)
Conv(128) → BN → ReLU → MaxPool → Dropout(0.25)
AdaptiveAvgPool → Dense(256) → BN → ReLU → Dropout(0.5) → Linear(num_classes)
```

### Transfer Learning: MobileNetV2

- **Backbone**: MobileNetV2 pre-trained on ImageNet (frozen → fine-tuned)
- **Head**: Dropout(0.3) → Dense(256) → BN → ReLU → Dropout(0.2) → Linear
- **Fine-tuning**: Top backbone layers unfrozen with reduced LR

### ⭐ Recommended: EfficientNet-B2

- **Backbone**: EfficientNet-B2 pre-trained on ImageNet (compound scaling)
- **Head**: Dropout(0.3) → Dense(512) → BN → ReLU → Dropout(0.2) → Linear
- **Fine-tuning**: Last ~37% of backbone unfrozen with CosineAnnealingLR
- **Why EfficientNet?** Compound scaling (depth + width + resolution) → significantly better features

---

## 🧪 Testing Instructions

### Verify Auto-Download

```bash
# 1. Delete cached model (if any)
rm -rf ~/.cache/huggingface/hub/models--Anuraaag17--mushroom-classifier-models

# 2. Run the app
streamlit run app/app.py

# 3. Verify: "Downloading model..." spinner should appear
# 4. Upload an image → prediction works
# 5. Re-run: no download spinner (cached)
```

### Verify Safety Layer

1. Upload a mushroom image
2. Set confidence threshold to 0.95 (to trigger low-confidence for testing)
3. Verify: if model predicts EDIBLE, it should be overridden to UNCERTAIN

### Verify No Heavy Files

```bash
# Should return nothing
git ls-files '*.pth' '*.pt' '*.h5' '*.ckpt'
```

---

## 🔥 Explainability: Grad-CAM

Grad-CAM produces visual explanations by computing gradients of the predicted class w.r.t. the last convolutional layer, producing heatmaps showing **which regions** the model focused on (cap texture, gill structure, stem shape).

---

## 📈 Training (Development Only)

```bash
# Install dev dependencies
pip install pandas scikit-learn matplotlib seaborn kagglehub

# Train EfficientNet-B2 (RECOMMENDED)
python -m src.train --model efficientnet --epochs 25 --fine_tune

# Evaluate
python -m src.evaluate --model_path models/efficientnet_b2_final.pth --model_type efficientnet
```

---

## 🔗 Links

- 📦 [GitHub Repository](https://github.com/anuraag45/mushroom-classifier)
- 🤗 [Model Weights (Hugging Face)](https://huggingface.co/Anuraaag17/mushroom-classifier-models)
- 🚀 [Live Demo (Streamlit Cloud)](https://mushroom-classifier.streamlit.app) *(deploy to activate)*

---

## 📄 License

Educational and research purposes only. Toxicity classifications are approximations and must not be used as the sole basis for consuming wild mushrooms.