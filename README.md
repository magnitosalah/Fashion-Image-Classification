# Fashion Product Image Classification: Addressing Long-Tail Data Imbalance

## 📌 Project Goal
The primary objective of this project is to develop a robust multi-class image classification model capable of accurately categorizing low-resolution fashion product images. Specifically, this project aims to solve the extreme **long-tail data imbalance problem** inherent in fashion datasets. By implementing an `EfficientNet-V2-S` architecture with transfer learning and integrating a **Class-Balanced Loss (Class-Weighted Loss)** strategy, we successfully improved the model's recall for minority classes without severely sacrificing overall global accuracy.

## 📊 Dataset Information
* **Dataset Name:** `ashraq/fashion-product-images-small` (Loaded via HuggingFace `datasets` library)[cite: 1, 5].
* **Task:** Multi-class image classification predicting the `articleType`[cite: 1, 5].
* **Classes:** 141 unique sub-categories[cite: 1, 5].
* **Data Split:** The dataset is split into Train, Validation, and Test sets using a strict **6:2:2 Stratified Split** to ensure minority classes are proportionally represented across all phases.

## 🛠 Environment and Dependencies
This project was developed and evaluated using PyTorch. 

**Core Dependencies:**
* `torch`, `torchvision` (Model architecture and training)[cite: 1, 5]
* `datasets` (HuggingFace dataset loading)[cite: 1, 5]
* `gradio` (Inference web demo)
* `matplotlib`, `seaborn` (Visualization of training curves and confusion matrices)[cite: 1, 5]
* `scikit-learn` (Metrics computation)[cite: 1, 5]
* `numpy`, `Pillow`[cite: 1, 3]

**Installation:**
```bash
pip install torch torchvision datasets gradio matplotlib seaborn scikit-learn numpy pillow
