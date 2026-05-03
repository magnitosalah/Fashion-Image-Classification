# Fashion Product Image Classification: Addressing Long-Tail Data Imbalance



## 📌 Project Goal

The primary objective of this project is to develop a robust multi-class image

classification model capable of accurately categorizing low-resolution fashion

product images, while solving the extreme long-tail data imbalance inherent in

the dataset. We employ **EfficientNet-V2-S** with transfer learning and integrate

a **Class-Weighted Loss** strategy to improve minority-class recall without

severely sacrificing global accuracy.



**Final Test Accuracy: 87.08%** | **Minority-Class Recall: ~0% → ~60%**



## 📊 Dataset Information

- **Dataset:** `ashraq/fashion-product-images-small` (HuggingFace `datasets`)

- **Task:** Multi-class classification of `articleType`

- **Classes:** 141 sub-categories

- **Size:** 44,072 images at 60×80 resolution

- **Split:** 6:2:2 Stratified Split (Train / Val / Test, fixed seed 42)



## 🛠 Environment and Dependencies

- Python ≥ 3.9, CUDA-enabled GPU recommended

- PyTorch, torchvision

- datasets (HuggingFace), gradio

- matplotlib, seaborn, scikit-learn, numpy, Pillow



```bash

pip install torch torchvision datasets gradio matplotlib seaborn scikit-learn numpy pillow

```



## 📁 Repository Structure

```

.

├── baseline_model.py          # Model 1: from-scratch EfficientNet-V2-S baseline

├── Transfer_model.py          # Phase 1: pretrained + augmentation (no weighted loss)

├── weight_changed_model.py    # Phase 2: weighted loss

├── final_model.py             # Phase 2/3: full pipeline (pretrained + aug + weighted loss)

├── demo_baseline.py           # Gradio demo for the baseline model

├── demo_final.py              # Gradio demo for the final model

├── improved_accuracy.py       # Top-10 most-improved minority class chart (Figure 7)

├── majority_accuracy.py       # Top-10 majority class trade-off chart (Figure 9)

├── bottom10_prediction.py     # Qualitative failure cases for bottom-10 rare classes (Figure 8)

├── correct_prediction.py      # Qualitative correct predictions on minority classes

├── scripts/                   # Shell scripts for automated grid search
│   ├── run_baseline.sh        # Trains the pure baseline model
│   ├── run_phase1.sh          # Phase 1 basic enhancements experiments
│   ├── run_phase2.sh          # Phase 2 class-weighted loss experiments
│   └── run_phase3.sh          # Phase 3 final hyperparameter optimization

├── plots/                     # Auto-saved training curves & confusion matrices

└── README.md

```



## 🚀 Training (Automated via Shell Scripts)
We provide shell scripts under the `scripts/` directory to easily reproduce all experiments and grid searches. Make sure to give execution permissions to the scripts if needed (`chmod +x scripts/*.sh`).

### Model 1: Baseline (Control Group)
Trains EfficientNet-V2-S from scratch without any enhancements.
```bash
bash scripts/run_baseline.sh
```
### Phase 1: Pretrained + Augmentation (No Weighted Loss)
Runs 4 configurations isolating the effects of basic enhancements (transfer learning and data augmentation).
```bash
bash bash scripts/run_phase1.sh 
```
### Phase 2: Class-Weighted Loss Integration
Runs the same 4 configurations as Phase 1, but with class-weighted loss applied to isolate its effect on resolving data imbalance.
```bash
bash scripts/run_phase2.sh
```
### Phase 3: Final Hyperparameter Optimization
Runs the extended epochs (30) and learning rate grid search using the SGD optimizer. The `lr=5e-4` configuration produces the **Final Model (87.08%)** reported in the paper.
```bash
bash scripts/run_phase3.sh
```

*(Note: `weight_changed_model.py` is included for custom weight experimentation and can be run manually if needed).*

## 📈 Evaluation

Each training script (`baseline_model.py`, `Transfer_model.py`, `final_model.py`)

automatically performs final test-set evaluation after training, and saves:

- Training curves → `plots/training_curves_<timestamp>.png`

- Confusion matrix (141×141) → `plots/confusion_matrix_<timestamp>.png`

- Per-class prediction bars (final model only) → `plots/per_class_prediction_bars_<timestamp>.png`

- CSV log → `training_log_<timestamp>.csv`



### Class-wise comparison plots (Baseline vs Final)

```bash

python improved_accuracy.py    # Top-10 most improved minority classes

python majority_accuracy.py    # Top-10 majority classes (trade-off analysis)

```



### Qualitative analysis

```bash

python bottom10_prediction.py  # Misclassification examples on rare classes

python correct_prediction.py   # Correct predictions on minority classes

```



## 🎨 Inference Demo

Launch a Gradio web interface to test the model on new images:



```bash

# Final model demo

python demo_final.py



# Baseline model demo (for comparison)

python demo_baseline.py

```

A local URL (`http://127.0.0.1:7860`) and a temporary public share link

will be printed. Drag & drop a fashion product image to see the top-5

predicted classes with confidence scores.



## 📂 Required Checkpoints
The demo and analysis scripts expect the following weights in the project root:

./baseline.pth — produced by scripts/run_baseline.sh

./efficientnet_v2s.pth — produced by scripts/run_phase3.sh


## 👥 Authors

- 20230349 Donggyu Yu

- 20220561 Jisu Kim

- 20220422 Seunghyun Lee
