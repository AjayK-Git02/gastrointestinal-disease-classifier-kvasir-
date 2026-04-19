# 🤝 Contributing to Gastrointestinal Disease Classifier

Thank you for your interest in contributing! This project welcomes improvements in model accuracy, code quality, and documentation.

---

## 🛠️ How to Contribute

### 1. Fork & Clone
```bash
git fork https://github.com/AjayK-Git02/gastrointestinal-disease-classifier-kvasir-.git
git clone https://github.com/<your-username>/gastrointestinal-disease-classifier-kvasir-.git
cd gastrointestinal-disease-classifier-kvasir-
```

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Set Up Environment
```bash
conda create -n kvasir_env python=3.8
conda activate kvasir_env
pip install -r requirements.txt
```

### 4. Make Changes & Test
- Ensure your changes run correctly in the notebook
- Add descriptive markdown cells explaining new techniques

### 5. Commit & Push
```bash
git add .
git commit -m "feat: add data augmentation pipeline"
git push origin feature/your-feature-name
```

### 6. Open a Pull Request
- Describe what you changed and why
- Reference any related issues

---

## 🎯 Priority Contribution Areas

| Area | Description |
|------|-------------|
| 🔴 Data Augmentation | Add flip, rotation, zoom, brightness augmentation |
| 🔴 Callbacks | Implement EarlyStopping + ReduceLROnPlateau |
| 🔴 Evaluation Metrics | Add confusion matrix, F1, precision, recall |
| 🟡 Fine-tuning | Unfreeze top VGG16 layers for fine-tuning |
| 🟡 Grad-CAM | Implement explainability visualization |
| 🟢 Alternative Models | Try EfficientNetB0, ResNet50, MobileNetV2 |

---

## 📝 Code Style Guidelines

- Use clear variable names
- Add comments for non-obvious logic
- Add markdown cells between notebook sections
- Set random seeds for reproducibility:
  ```python
  import random, numpy as np, tensorflow as tf
  random.seed(42); np.random.seed(42); tf.random.set_seed(42)
  ```

---

## ⚖️ License

By contributing, you agree your contributions will be licensed under the [MIT License](LICENSE).
