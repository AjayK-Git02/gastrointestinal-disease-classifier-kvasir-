<div align="center">

# 🏥 Gastrointestinal Disease Classifier

### Deep Learning for Endoscopic Image Analysis using VGG16 Transfer Learning on the Kvasir Dataset

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kvasir%20v1-8B5CF6?style=for-the-badge)](https://datasets.simula.no/kvasir/)
[![Val Accuracy](https://img.shields.io/badge/Val%20Accuracy-82.5%25-F59E0B?style=for-the-badge)]()

<br/>

> **⚕️ Disclaimer:** This model is for **research and educational purposes only** and is **not intended for clinical use**.

</div>

---

## 📦 Model Download

> The pre-trained model (`kvasir_vgg16_model.h5`, ~136 MB) is **not stored in this repo** due to GitHub's 100 MB file size limit.  
> ➡️ **Download it here:** *(Upload to Google Drive / Kaggle and update this link)*

---

## 📋 Table of Contents

- [🔍 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🧠 Model Architecture](#-model-architecture)
- [📈 Training Results](#-training-results)
- [🚀 Getting Started](#-getting-started)
- [💻 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [⚠️ Limitations & Future Work](#️-limitations--future-work)
- [📚 References](#-references)
- [👤 Author](#-author)

---

## 🔍 Project Overview

This project develops a **deep learning image classifier** to automatically detect and categorize **gastrointestinal (GI) diseases** from endoscopic images. It leverages **VGG16 transfer learning** (pre-trained on ImageNet) with a custom classification head fine-tuned for **8 GI-related classes** from the Kvasir Dataset.

### 🎯 Key Goals
- Assist medical professionals in identifying GI tract conditions from endoscopic images
- Demonstrate the effectiveness of transfer learning for medical image classification
- Provide a reproducible baseline for further research and experimentation

### 🏆 Key Achievements
| Metric | Value |
|--------|-------|
| Final Validation Accuracy | **82.5%** |
| Training Accuracy | **88.6%** |
| Classes Classified | **8** |
| Total Training Images | **6,400** |
| Model Size | **~136 MB** |

---

## 📊 Dataset

The **[Kvasir Dataset v1](https://datasets.simula.no/kvasir/)** (Simula Research Laboratory) contains 8,000 endoscopic images across 8 labeled classes, licensed under **CC BY 4.0**.

### 🗂️ Classes

| # | Class | Type | Description |
|---|-------|------|-------------|
| 1 | `dyed-lifted-polyps` | 🔴 Pathological | Polyps marked with dye and mechanically lifted |
| 2 | `dyed-resection-margins` | 🟠 Post-procedural | Resection margins marked with dye post-removal |
| 3 | `esophagitis` | 🔴 Pathological | Inflammation of the esophagus lining |
| 4 | `normal-cecum` | 🟢 Normal | Healthy cecum tissue |
| 5 | `normal-pylorus` | 🟢 Normal | Healthy pylorus tissue |
| 6 | `normal-z-line` | 🟢 Normal | Normal gastroesophageal junction (Z-line) |
| 7 | `polyps` | 🔴 Pathological | Gastrointestinal polyp growths |
| 8 | `ulcerative-colitis` | 🔴 Pathological | Chronic inflammatory condition causing ulcers |

### ✂️ Data Split

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 6,400 | 80% |
| Validation | 1,600 | 20% |
| **Total** | **8,000** | **100%** |

### ⚙️ Preprocessing Pipeline

```
Raw Image (variable size)
    │
    ▼
Resize to 224×224 pixels
    │
    ▼
Normalize pixel values: [0, 255] → [0.0, 1.0]
    │
    ▼
Batch (size = 32)
    │
    ▼
Feed to VGG16
```

> ⚠️ **Note:** No data augmentation was applied in this baseline run.

---

## 🧠 Model Architecture

### Network Diagram

```
Input (224×224×3)
    │
    ▼
┌─────────────────────────────────────┐
│         VGG16 Base Model            │
│      (ImageNet Weights, Frozen)     │
│      ~14.7M parameters (frozen)     │
└─────────────────────────────────────┘
    │
    ▼
Flatten() — Convert feature maps to 1D
    │
    ▼
Dense(256, activation='relu')
    │
    ▼
Dropout(0.5) — Regularization
    │
    ▼
Dense(8, activation='softmax') ← Output: 8 GI classes
```

### ⚙️ Compilation Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | `Adam` | Adaptive learning, robust convergence |
| Learning Rate | `0.0001` | Small LR to avoid disrupting pretrained features |
| Loss Function | `Categorical Crossentropy` | Standard for multi-class classification |
| Metric | `Accuracy` | Primary evaluation metric |

### 🔑 Transfer Learning Strategy

| Component | Status | Parameters |
|-----------|--------|-----------|
| VGG16 Conv Layers | ❄️ Frozen | ~14.7M (not updated) |
| Custom Dense Layers | 🔥 Trainable | ~524K (updated during training) |

---

## 📈 Training Results

Training over **10 Epochs** on CPU (~27 min/epoch, ~4.5 hours total):

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Notes |
|-------|-----------|-----------|---------|---------|-------|
| 1 | 1.0570 | 57.84% | 0.7292 | 71.13% | 🚀 Fast initial convergence |
| 2 | 0.6855 | 72.64% | 0.5866 | 77.00% | |
| 3 | 0.5604 | 78.42% | 0.5475 | 79.00% | |
| 4 | 0.4978 | 80.77% | 0.4908 | 80.81% | |
| 5 | 0.4351 | 83.23% | 0.4869 | 80.62% | ⚠️ Val acc dip |
| 6 | 0.4033 | 83.91% | 0.4458 | 82.69% | ✅ Recovery |
| 7 | 0.3713 | 85.59% | 0.4831 | 80.81% | ⚠️ Val acc dip |
| 8 | 0.3430 | 86.67% | 0.4361 | 82.19% | |
| 9 | 0.3086 | 88.55% | 0.4346 | 82.19% | |
| **10** | **0.3004** | **88.58%** | **0.4226** | **82.50%** | 🏆 Best |

### 📊 Performance Analysis

```
Training Accuracy   ████████████████████████████████████████ 88.6%
Validation Accuracy ████████████████████████████████▌        82.5%
                    0%                                       100%
```

- **Accuracy Gap (6.1%)** → Mild overfitting; train/val divergence after epoch 4
- **Val Accuracy Plateau** → Suggests training without callbacks is suboptimal
- **Fast Early Convergence** → Transfer learning from ImageNet is highly effective

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `conda` or `pip`
- ~8 GB RAM recommended
- GPU optional but strongly recommended (CPU training: ~27 min/epoch)

### ⚡ Quick Installation

```bash
# 1. Clone the repository
git clone https://github.com/AjayK-Git02/gastrointestinal-disease-classifier-kvasir-.git
cd gastrointestinal-disease-classifier-kvasir-

# 2. Create and activate conda environment
conda create -n kvasir_env python=3.8 -y
conda activate kvasir_env

# 3. Install all dependencies
pip install -r requirements.txt
```

### 📂 Dataset Setup

1. Download from [datasets.simula.no/kvasir](https://datasets.simula.no/kvasir/)
2. Extract and place class folders in the project root:

```
gastrointestinal-disease-classifier-kvasir-/
├── dyed-lifted-polyps/          ← 1000 images
├── dyed-resection-margins/      ← 1000 images
├── esophagitis/                 ← 1000 images
├── normal-cecum/                ← 1000 images
├── normal-pylorus/              ← 1000 images
├── normal-z-line/               ← 1000 images
├── polyps/                      ← 1000 images
├── ulcerative-colitis/          ← 1000 images
└── kvasir_vgg16_classifier.ipynb
```

> ⚠️ **Important:** Remove `.ipynb_checkpoints` if it exists inside your dataset folder — it will be detected as an extra class:
> ```bash
> find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} +
> ```

---

## 💻 Usage

### ▶️ Run the Training Notebook

```bash
jupyter notebook kvasir_vgg16_classifier.ipynb
```

### 🔮 Run Inference on a Custom Image

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Class labels (alphabetical — matches flow_from_directory)
CLASS_NAMES = [
    'dyed-lifted-polyps',
    'dyed-resection-margins',
    'esophagitis',
    'normal-cecum',
    'normal-pylorus',
    'normal-z-line',
    'polyps',
    'ulcerative-colitis'
]

def predict_image(img_path, model_path="kvasir_vgg16_model.h5"):
    """Run inference on a single endoscopic image."""
    model = load_model(model_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    probs = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(probs)]
    confidence = np.max(probs) * 100

    print(f"🏷️  Predicted Class : {predicted_class}")
    print(f"📊 Confidence      : {confidence:.2f}%")
    print(f"\n📈 All Class Probabilities:")
    for name, prob in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {name:<28} {bar} {prob*100:.2f}%")

predict_image("path/to/your/endoscopy_image.jpg")
```

---

## 📁 Project Structure

```
gastrointestinal-disease-classifier-kvasir-/
│
├── 📓 kvasir_vgg16_classifier.ipynb   # Main training & evaluation notebook
├── 📋 README.md                        # Project documentation (this file)
├── 📄 TDR.md                           # Technical Design Report
├── 📦 requirements.txt                 # Python dependencies
├── 🤝 CONTRIBUTING.md                  # Contribution guidelines
├── ⚖️  LICENSE                          # MIT License
├── 🙈 .gitignore                       # Git ignore rules
├── 🗒️  .gitattributes                   # Git LFS / line ending config
│
└── 🚫 [NOT IN REPO — too large]
    ├── kvasir_vgg16_model.h5           # Trained model (~136 MB)
    ├── dyed-lifted-polyps/             # Dataset images
    ├── dyed-resection-margins/
    ├── esophagitis/
    ├── normal-cecum/
    ├── normal-pylorus/
    ├── normal-z-line/
    ├── polyps/
    └── ulcerative-colitis/
```

---

## ⚠️ Limitations & Future Work

### 🐛 Known Issues

| Issue | Status | Fix |
|-------|--------|-----|
| `.ipynb_checkpoints` detected as extra class | ⚠️ | Delete before training |
| Model saved in deprecated `.h5` format | ⚠️ | Use `.keras` format |
| Deprecated TF 2.15 API warnings | ℹ️ | Cosmetic only |
| No training callbacks | ❌ | Add EarlyStopping + ReduceLROnPlateau |

### 🔮 Roadmap

| Priority | Improvement | Expected Impact |
|----------|-------------|-----------------|
| 🔴 High | Add **data augmentation** (flip, rotation, zoom, brightness) | +2-5% accuracy |
| 🔴 High | Add **EarlyStopping + ReduceLROnPlateau callbacks** | Prevent overfitting |
| 🔴 High | **Confusion matrix + F1/Precision/Recall** evaluation | Better insight |
| 🟡 Medium | **Fine-tune top VGG16 layers** (unfreeze last 4 layers) | +3-5% accuracy |
| 🟡 Medium | **Grad-CAM visualization** for model explainability | Clinical trust |
| 🟡 Medium | **Weighted loss** for class imbalance | Better minority class perf |
| 🟢 Low | Try **EfficientNetB0 / ResNet50 / MobileNetV2** | Architecture comparison |
| 🟢 Low | Export to **TensorFlow Lite** for mobile deployment | Deployment-ready |

### 💡 Quick Model Improvements (Code Snippets)

<details>
<summary><strong>Add Data Augmentation</strong></summary>

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.1,
    height_shift_range=0.1
)
```
</details>

<details>
<summary><strong>Add Training Callbacks</strong></summary>

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(..., callbacks=callbacks)
```
</details>

<details>
<summary><strong>Fine-tune Top VGG16 Layers</strong></summary>

```python
# Unfreeze last 4 layers of VGG16
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile with a much lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```
</details>

<details>
<summary><strong>Evaluation with Confusion Matrix & Classification Report</strong></summary>

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = np.argmax(model.predict(val_generator), axis=1)
y_true = val_generator.classes

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
```
</details>

---

## 📚 References

| Resource | Link |
|----------|------|
| Kvasir Dataset Paper | [Pogorelov et al., 2017 (ACM)](https://dl.acm.org/doi/10.1145/3193025.3193037) |
| VGG16 Paper | [Simonyan & Zisserman, 2014 (arXiv)](https://arxiv.org/abs/1409.1556) |
| TensorFlow Documentation | [tensorflow.org](https://www.tensorflow.org/api_docs/python/tf/keras) |
| Kvasir Dataset Download | [datasets.simula.no/kvasir](https://datasets.simula.no/kvasir/) |

---

## 👤 Author

<div align="center">

**Ajay K**

[![GitHub](https://img.shields.io/badge/GitHub-AjayK--Git02-181717?style=for-the-badge&logo=github)](https://github.com/AjayK-Git02)

</div>

---

<div align="center">

⭐ **If this project helped you, please consider giving it a star!** ⭐

*Built with ❤️ for medical AI research*

</div>
