# Technical Design Report (TDR)
## Gastrointestinal Disease Classifier — VGG16 + Kvasir Dataset

**Author:** Ajay K  
**Date:** April 2026  
**Version:** 1.0  
**Repository:** [AjayK-Git02/gastrointestinal-disease-classifier-kvasir-](https://github.com/AjayK-Git02/gastrointestinal-disease-classifier-kvasir-)

---

## 1. Project Summary

This document provides a technical design review of a **deep learning image classification system** built to identify gastrointestinal (GI) tract conditions from endoscopic images. The system uses **VGG16 transfer learning** trained on the **Kvasir Dataset** and is implemented in a Jupyter Notebook using TensorFlow/Keras.

---

## 2. Problem Statement

Gastrointestinal diseases affect millions of people globally. Endoscopy is the primary diagnostic tool, but interpretation is time-consuming and requires specialist expertise. Automating the classification of endoscopic images can:

- Assist clinicians in prioritizing cases
- Reduce human error in high-workload environments
- Enable screening at scale in under-resourced settings

**Target:** Build a multi-class image classifier capable of distinguishing 8 GI conditions (+ normal variants) from endoscopic images.

---

## 3. Dataset

### 3.1 Source
- **Name:** Kvasir Dataset v1 (Multi-Class)
- **Origin:** Simula Research Laboratory
- **License:** CC BY 4.0
- **URL:** https://datasets.simula.no/kvasir/

### 3.2 Classes (8 Classes Total)

| # | Class Label | Type |
|---|------------|------|
| 1 | `dyed-lifted-polyps` | Pathological |
| 2 | `dyed-resection-margins` | Post-procedural |
| 3 | `esophagitis` | Pathological |
| 4 | `normal-cecum` | Normal |
| 5 | `normal-pylorus` | Normal |
| 6 | `normal-z-line` | Normal |
| 7 | `polyps` | Pathological |
| 8 | `ulcerative-colitis` | Pathological |

> ⚠️ The notebook output lists 9 classes from data generator (`9 class_indices`) — this may include a misc folder (e.g., `.ipynb_checkpoints`) picked up during directory scan. The actual Kvasir v1 dataset has **8 labelled classes**.

### 3.3 Data Split

| Split | Images | Percentage |
|-------|--------|-----------|
| Training | 6,400 | 80% |
| Validation | 1,600 | 20% |
| **Total** | **8,000** | **100%** |

Split performed using `ImageDataGenerator(validation_split=0.2)`.

### 3.4 Preprocessing

| Step | Details |
|------|---------|
| Rescaling | Divide pixel values by 255 → range `[0.0, 1.0]` |
| Resize | All images resized to `(224, 224)` pixels |
| Batch Size | 32 |
| Augmentation | ❌ None applied |

---

## 4. Model Architecture

### 4.1 Base Model: VGG16

VGG16 (Simonyan & Zisserman, 2014) is a 16-layer deep CNN originally trained on ImageNet (1.2M images, 1000 classes). It captures rich spatial features via stacked 3×3 convolution layers.

**Settings Used:**
- `weights='imagenet'` — Pre-trained weights loaded
- `include_top=False` — Final classification head excluded
- `input_shape=(224, 224, 3)` — RGB images at standard resolution

**Parameters:** ~14.7 million (all frozen)

### 4.2 Custom Classification Head

```
VGG16 base (frozen)
└── Flatten()
└── Dense(256, activation='relu')
└── Dropout(0.5)
└── Dense(9, activation='softmax')  ← Output layer
```

**Rationale for choices:**
- **Flatten:** Converts convolutional feature maps to 1D vector
- **Dense(256):** Learns task-specific feature combinations
- **Dropout(0.5):** Regularization to reduce overfitting
- **Dense(9, softmax):** Multi-class probability output

### 4.3 Compilation

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Categorical Crossentropy |
| Metric | Accuracy |

---

## 5. Training Protocol

### 5.1 Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Steps/Epoch | 200 (derived from 6400 / 32) |
| Validation Steps | 50 (derived from 1600 / 32) |
| Environment | CPU only (no GPU detected) |
| Approx. Time/Epoch | ~27 minutes |
| Total Training Time | ~4.5 hours |

### 5.2 Training History

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Notes |
|-------|-----------|-----------|---------|---------|-------|
| 1 | 1.0570 | 57.84% | 0.7292 | 71.13% | Fast convergence |
| 2 | 0.6855 | 72.64% | 0.5866 | 77.00% | |
| 3 | 0.5604 | 78.42% | 0.5475 | 79.00% | |
| 4 | 0.4978 | 80.77% | 0.4908 | 80.81% | |
| 5 | 0.4351 | 83.23% | 0.4869 | 80.62% | ⚠️ Val acc drop |
| 6 | 0.4033 | 83.91% | 0.4458 | 82.69% | Recovery |
| 7 | 0.3713 | 85.59% | 0.4831 | 80.81% | ⚠️ Val acc drop again |
| 8 | 0.3430 | 86.67% | 0.4361 | 82.19% | |
| 9 | 0.3086 | 88.55% | 0.4346 | 82.19% | |
| 10 | 0.3004 | 88.58% | 0.4226 | 82.50% | Best val acc |

### 5.3 Observations

1. **Training accuracy** grew steadily from 57.8% → 88.6%
2. **Validation accuracy** plateaued around 82-83% from epoch 4 onward
3. **Accuracy gap** (Train: 88.6% vs Val: 82.5%) suggests mild **overfitting**
4. Epochs 5 and 7 show validation accuracy dips, indicating training instability without callbacks

---

## 6. Saved Model

| Property | Value |
|----------|-------|
| File | `kvasir_vgg16_model.h5` |
| Format | HDF5 (legacy Keras format) |
| Size | ~136 MB |
| Warning | Keras 3.x recommends `.keras` native format instead |

---

## 7. Inference Testing

A single test was performed on a polyp image:

| Item | Value |
|------|-------|
| Test Image | `polyps/polyps/00072d5f-7cd8-434c-8a5a-1a0bb2c9711d.jpg` |
| Ground Truth | `polyps` |
| Predicted Class | `dyed-lifted-polyps` |
| Confidence | 50.85% |

**Result: ❌ Misclassified**

**Raw probabilities:**
```
[0.0001, 0.5085, 0.0016, 0.0008, 0.0004, 0.1787, 0.0018, 0.3048, 0.0034]
```

The model confused `polyps` (index 6) with `dyed-lifted-polyps` (index 1) — visually similar classes. The second-highest probability (30.48%) was assigned to `ulcerative-colitis`, showing uncertain boundary conditions between pathological categories.

---

## 8. Technical Issues Identified

### 8.1 Warnings

| Warning | Cause | Impact |
|---------|-------|--------|
| `tf.losses.sparse_softmax_cross_entropy deprecated` | TF 2.15 API change | Low — cosmetic |
| `tf.executing_eagerly_outside_functions deprecated` | TF 2.15 API change | Low — cosmetic |
| `tf.nn.max_pool deprecated` | TF 2.15 API change | Low — cosmetic |
| `HDF5 (.h5) model format deprecated` | Keras 3.x migration | Medium — use `.keras` |

### 8.2 Potential Data Issue

The `flow_from_directory()` detected **9 classes** but Kvasir v1 has only 8. The extra "class" is likely the `.ipynb_checkpoints/` folder. This should be explicitly excluded.

**Fix:**
```python
import shutil, os
checkpoints = os.path.join(dataset_path, '.ipynb_checkpoints')
if os.path.exists(checkpoints):
    shutil.rmtree(checkpoints)
```

### 8.3 No Callbacks Defined

No training callbacks were configured. This means:
- Training runs for full 10 epochs regardless of stagnation
- No learning rate decay when validation plateaus
- Risk of wasted compute / overfitting

---

## 9. What You Should Do Next

### ✅ Immediate Fixes

1. **Remove `.ipynb_checkpoints`** from dataset path before training
2. **Save model in native Keras format**:
   ```python
   model.save("kvasir_vgg16_model.keras")  # Instead of .h5
   ```

### 🔧 Model Improvements

3. **Add Data Augmentation:**
   ```python
   datagen = ImageDataGenerator(
       rescale=1./255,
       validation_split=0.2,
       rotation_range=20,
       horizontal_flip=True,
       zoom_range=0.2,
       brightness_range=[0.8, 1.2]
   )
   ```

4. **Add Callbacks:**
   ```python
   from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
   callbacks = [
       EarlyStopping(patience=3, restore_best_weights=True),
       ReduceLROnPlateau(factor=0.5, patience=2)
   ]
   history = model.fit(..., callbacks=callbacks)
   ```

5. **Fine-tune VGG16 top layers (optional):**
   ```python
   # Unfreeze last 4 layers of VGG16
   for layer in base_model.layers[-4:]:
       layer.trainable = True
   # Recompile with lower LR
   model.compile(optimizer=Adam(1e-5), ...)
   ```

6. **Add Confusion Matrix & Classification Report:**
   ```python
   from sklearn.metrics import classification_report, confusion_matrix
   y_pred = np.argmax(model.predict(val_data), axis=1)
   y_true = val_data.classes
   print(classification_report(y_true, y_pred, target_names=class_names))
   ```

7. **Implement Grad-CAM** for explainability (visualizes which regions the model focuses on)

8. **Try alternative architectures**: EfficientNetB0, ResNet50, MobileNetV2

---

## 10. Code Quality Assessment

| Area | Score | Notes |
|------|-------|-------|
| Code Clarity | ✅ Good | Clean, step-by-step cells |
| Documentation | ⚠️ Minimal | No markdown cells explaining steps |
| Error Handling | ❌ Missing | No try/except or path validation |
| Reproducibility | ⚠️ Partial | No random seed set |
| Callbacks | ❌ Missing | No early stopping / LR scheduling |
| Metrics | ⚠️ Partial | Only accuracy tracked; no precision/recall |
| Model Saving | ⚠️ Legacy format | Use `.keras` format |

**Suggested Seed Fix:**
```python
import random, numpy as np, tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

---

## 11. Environment

| Component | Version |
|-----------|---------|
| Python | 3.x (conda env: `kvasir_env`) |
| TensorFlow | 2.15.0 |
| Keras | Bundled with TF 2.15 |
| NumPy | Latest compatible |
| Matplotlib | Latest compatible |
| Hardware | CPU (no GPU detected) |

---

## 12. Summary & Recommendations

| Category | Recommendation | Priority |
|----------|---------------|---------|
| Data | Remove `.ipynb_checkpoints` from dataset | 🔴 High |
| Data | Add augmentation | 🟡 Medium |
| Model | Add callbacks | 🔴 High |
| Model | Fine-tune top VGG16 layers | 🟡 Medium |
| Model | Try EfficientNet or ResNet50 | 🟢 Low |
| Evaluation | Add confusion matrix & F1 scores | 🔴 High |
| Explainability | Implement Grad-CAM | 🟡 Medium |
| Code | Set random seeds | 🟡 Medium |
| Deployment | Convert to TFLite for mobile use | 🟢 Low |

**Current Status:** Functional baseline model with ~82.5% validation accuracy. The model demonstrates the viability of VGG16 transfer learning for GI disease classification, but requires further engineering to be production-ready.

---

*TDR generated on: April 14, 2026*  
*Project: Gastrointestinal Disease Classifier — Kvasir + VGG16*
