# 🍄 Smart Mushroom Classification & Toxicity Risk Assessment

A production-grade computer vision system that classifies mushroom images into species and predicts whether they are **edible** or **poisonous**, complete with Grad-CAM visual explainability and a Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?logo=streamlit)

---

## 🎯 Problem Statement

Misidentifying poisonous mushrooms kills hundreds of people worldwide every year. This project builds an AI-powered safety tool that:

1. **Classifies** mushroom images into species (multi-class classification)
2. **Predicts** edibility (binary: EDIBLE vs POISONOUS)
3. **Warns** users when prediction confidence is low
4. **Explains** decisions visually using Grad-CAM heatmaps

> ⚠️ **Disclaimer**: This tool is for educational purposes only. Never consume wild mushrooms based solely on AI predictions.

---

## 🛠️ Tech Stack

| Component        | Technology                                              |
|------------------|---------------------------------------------------------|
| ML Framework     | PyTorch / torchvision                                   |
| Models           | Custom CNN · MobileNetV2 · **EfficientNet-B2** (recommended) |
| Training         | AMP (Mixed Precision) · AdamW · OneCycleLR · Label Smoothing |
| Data Pipeline    | torchvision transforms + DataLoader                     |
| Explainability   | Grad-CAM (hook-based)                                   |
| Web App          | Streamlit                                               |
| Evaluation       | scikit-learn, Seaborn, Matplotlib                       |
| Dataset Source   | Kaggle via `kagglehub`                                  |

---

## 📁 Project Structure

```
mushroom-classifier/
├── app/
│   └── app.py                  # Streamlit web application
├── data/                       # Dataset (auto-downloaded)
├── models/                     # Saved trained models (.pth)
├── notebooks/
│   └── eda.py                  # Exploratory Data Analysis script
├── outputs/                    # Plots, metrics, confusion matrices
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Download, cleaning, splits, augmentation
│   ├── model.py                # CNN, MobileNetV2, and EfficientNet-B2 architectures
│   ├── train.py                # Training loop with AMP, OneCycleLR, label smoothing
│   ├── evaluate.py             # Metrics, confusion matrix, comparison
│   ├── predict.py              # Single & batch inference with risk warnings
│   └── explainability.py       # Grad-CAM heatmap generation
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd mushroom-classifier
pip install -r requirements.txt
```

### 2. Run EDA (Optional)

```bash
python notebooks/eda.py
```

### 3. Train Models

```bash
# Train EfficientNet-B2 (RECOMMENDED — best accuracy)
python -m src.train --model efficientnet --epochs 25 --fine_tune

# Train MobileNetV2 with fine-tuning
python -m src.train --model transfer --epochs 20 --fine_tune

# Train custom CNN baseline
python -m src.train --model cnn --epochs 30
```

### 4. Evaluate & Compare

```bash
# Evaluate a single model
python -m src.evaluate --model_path models/efficientnet_b2_final.pth --model_type efficientnet

# Compare all available models
python -m src.evaluate --compare
```

### 5. Single / Batch Prediction

```bash
python -m src.predict --image path/to/mushroom.jpg
python -m src.predict --batch_dir path/to/images/
```

### 6. Launch Web App

```bash
streamlit run app/app.py
```

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
- **Why EfficientNet?** Compound scaling (depth + width + resolution) → significantly better features than MobileNetV2

---

## ⚡ Training Optimizations

| Technique | Purpose |
|-----------|---------|
| AMP (Mixed Precision) | 2× faster training on CUDA with lower memory usage |
| AdamW Optimizer | Better weight decay regularization than Adam |
| OneCycleLR | Super-convergence — faster and higher accuracy |
| CosineAnnealingLR | Smooth fine-tuning with gradual LR decay |
| Label Smoothing (0.1) | Prevents overconfident predictions, improves generalization |
| Gradient Clipping | Prevents exploding gradients during training |

---

## 🔥 Explainability: Grad-CAM

Grad-CAM produces visual explanations by computing gradients of the predicted class w.r.t. the last convolutional layer, producing heatmaps showing **which regions** the model focused on (cap texture, gill structure, stem shape).

---

## ⚠️ Risk-Based Output

```
if confidence < 70%:
    ⚠️ "Low confidence prediction. Do NOT rely on this result."
```

False negatives (EDIBLE when actually POISONOUS) can be fatal. The system defaults unknown species to POISONOUS and issues safety warnings when uncertain.

---

## 📈 Data Augmentation

| Augmentation       | Purpose                              |
|--------------------|--------------------------------------|
| Random Flip        | Handle different camera orientations |
| Random Rotation    | Account for rotated specimens        |
| RandomResizedCrop  | Vary subject scale                   |
| ColorJitter        | Simulate lighting variations         |
| GaussianBlur       | Simulate slight camera defocus       |

---

## 🔮 Future Improvements

- [ ] Cloud deployment (Docker + AWS/GCP)
- [ ] ONNX/TorchScript export for mobile
- [ ] Expand to 50+ species
- [ ] GPS-based species filtering
- [ ] Monte Carlo Dropout for uncertainty quantification

---

## 📄 License

Educational and research purposes only. Toxicity classifications are approximations and must not be used as the sole basis for consuming wild mushrooms.