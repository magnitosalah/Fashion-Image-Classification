import argparse
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset
from datetime import datetime
from datasets import load_dataset
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


# AUTO-INJECTED: Korean font setup for matplotlib
import os as _os
import matplotlib.font_manager as _fm
import matplotlib.pyplot as _plt
if not any('NanumGothic' in f.name for f in _fm.fontManager.ttflist):
    for _font in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                  '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf']:
        if _os.path.exists(_font):
            _fm.fontManager.addfont(_font)
_plt.rcParams.update({'font.family': 'NanumGothic', 'axes.unicode_minus': False})
del _os, _fm, _plt
# END AUTO-INJECTED Korean font setup



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--model_path", type=str, default="./efficientnet_v2s.pth")
    return parser.parse_args()


class FashionDataset(Dataset):
    """
    HuggingFace dataset을 감싸는 PyTorch Dataset.
    augment=True이면 train용 augmentation을 적용하고,
    augment=False이면 기본 전처리(preprocess)만 적용합니다.
    """
    def __init__(self, hf_dataset, indices, label_to_idx, preprocess, augment=False):
        self.hf_dataset   = hf_dataset
        self.indices      = indices
        self.label_to_idx = label_to_idx
        self.preprocess   = preprocess
        self.augment      = augment

        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.hf_dataset[self.indices[idx]]
        img    = sample["image"].convert("RGB")

        if self.augment:
            img = self.aug_transform(img)

        img   = self.preprocess(img)
        label = self.label_to_idx[sample["articleType"]]
        return img, label


def stratified_split(labels, train_ratio=0.6, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in class_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n       = len(indices)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train_idx.extend(indices[:n_train].tolist())
        val_idx.extend(indices[n_train:n_train + n_val].tolist())
        test_idx.extend(indices[n_train + n_val:].tolist())

    return train_idx, val_idx, test_idx


def compute_class_weights(train_idx, hf_dataset, label_to_idx, num_classes, device):
    """
    Train set의 클래스별 샘플 수를 기반으로 class weight를 계산합니다.

    공식: weight_c = total_samples / (num_classes * count_c)
      - 샘플이 적은 클래스일수록 높은 가중치를 부여합니다.
      - 샘플이 전혀 없는 클래스는 weight=0으로 처리합니다.
    """
    class_counts = np.zeros(num_classes, dtype=np.float32)
    for idx in train_idx:
        label = label_to_idx[hf_dataset[idx]["articleType"]]
        class_counts[label] += 1

    total = class_counts.sum()
    class_weights = np.where(
        class_counts > 0,
        total / (num_classes * class_counts),
        0.0
    )

    # 콘솔에 클래스별 weight 요약 출력
    nonzero = class_weights[class_weights > 0]
    print(f"Class Weights — min: {nonzero.min():.4f} / max: {nonzero.max():.4f} / mean: {nonzero.mean():.4f}")

    return torch.tensor(class_weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    return total_loss / total, correct / total


def evaluate_with_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images   = images.to(device)
            outputs  = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def save_training_curves(history, timestamp):
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)

    epochs         = range(1, len(history["train_loss"]) + 1)
    final_test_acc = history["final_test_acc"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Training Curves — {timestamp}", fontsize=16, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Accuracy")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]],   "r-o", label="Val Accuracy")
    axes[1].axhline(y=final_test_acc * 100, color="green", linestyle="--",
                    linewidth=2, label=f"Final Test Acc ({final_test_acc*100:.2f}%)")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = f"{plot_dir}/training_curves_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to: {path}")


def save_confusion_matrix(all_labels, all_preds, idx_to_label, timestamp):
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)

    class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    num_classes = len(class_names)
    cm          = confusion_matrix(all_labels, all_preds, labels=sorted(idx_to_label.keys()))

    # ── 핵심 변경: 가로축(예측 클래스) 기준 비율 계산 ──
    # cm의 각 열(column)을 해당 열의 합으로 나눔
    # → 색상은 "해당 예측 클래스 중 실제 클래스별 비율"을 표현
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_ratio = cm / row_sums                                  # 열 기준 정규화 (0.0 ~ 1.0)

    # annot용 텍스트: "카운트\n(비율%)" 형식
    annot_text = np.array([
        [f"{cm[i, j]}\n({cm_ratio[i, j]*100:.1f}%)" for j in range(num_classes)]
        for i in range(num_classes)
    ])

    fig_size  = max(20, int(num_classes * 0.35))
    font_size = max(5,  int(10 - num_classes * 0.03))

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_ratio,                                               # ← 색상 기준: 열 정규화 비율
        annot=annot_text, fmt="",                               # ← 텍스트: 카운트 + 비율
        cmap="Blues",
        vmin=0.0, vmax=1.0,                                     # 색상 범위 고정
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.3, linecolor="gray",
        annot_kws={"size": font_size}, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {timestamp}", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=font_size)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,              fontsize=font_size)

    plt.tight_layout()
    path = f"{plot_dir}/confusion_matrix_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to: {path}")


def save_per_class_prediction_bars(all_labels, all_preds, idx_to_label, timestamp):
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)
    class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    num_classes = len(class_names)
    cm          = confusion_matrix(all_labels, all_preds, labels=sorted(idx_to_label.keys()))
    # 행 기준 정규화 (각 실제 클래스 내 예측 비율)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_ratio = cm / row_sums  # (num_classes, num_classes)
    cols     = 4
    rows     = (num_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes      = axes.flatten()
    fig.suptitle(f"Per-Class Prediction Distribution — {timestamp}", fontsize=16, fontweight="bold")
    
    for i in range(num_classes):
        ax     = axes[i]
        ratios = cm_ratio[i]                          # i번째 실제 클래스의 예측 분포
        colors = ["steelblue"] * num_classes
        colors[i] = "tomato"                           # 정답(대각선) 강조
        bars = ax.bar(range(num_classes), ratios * 100, color=colors)
        ax.set_title(f"True: {class_names[i]}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("Ratio (%)", fontsize=7)
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_ylim(0, 110)
        ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.5)
        # 상위 3개 막대에만 수치 표시 (가독성)
        top3 = np.argsort(ratios)[-3:]
        for j in top3:
            if ratios[j] > 0:
                ax.text(j, ratios[j] * 100 + 1.5, f"{ratios[j]*100:.1f}%",
                        ha="center", va="bottom", fontsize=6)
                
    # 남는 서브플롯 숨기기
    for i in range(num_classes, len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    path = f"{plot_dir}/per_class_prediction_bars_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-class prediction bars saved to: {path}")


def main():
    args = get_args()

    print("===== Input Arguments =====")
    for k, v in vars(args).items():
        print(f"{k:15s}: {v}")
    print("===========================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 모델 및 전처리 설정 =====
    weights    = EfficientNet_V2_S_Weights.DEFAULT
    model      = efficientnet_v2_s(weights=weights)
    preprocess = weights.transforms()

    # ===== 데이터셋 로드 =====
    ds_full       = load_dataset("ashraq/fashion-product-images-small", split="train")
    unique_labels = sorted(set(ds_full["articleType"]))
    label_to_idx  = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label  = {idx: label for label, idx in label_to_idx.items()}
    num_classes   = len(unique_labels)
    all_labels    = [label_to_idx[lbl] for lbl in ds_full["articleType"]]

    # ===== Stratified Split (6:2:2) =====
    train_idx, val_idx, test_idx = stratified_split(all_labels, train_ratio=0.6, val_ratio=0.2, seed=42)
    print(f"Dataset Split => Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print("Data Augmentation: Applied to train set only (HorizontalFlip / Rotation15 / ColorJitter)\n")

    # ===== Dataset & DataLoader =====
    train_dataset = FashionDataset(ds_full, train_idx, label_to_idx, preprocess, augment=True)
    val_dataset   = FashionDataset(ds_full, val_idx,   label_to_idx, preprocess, augment=False)
    test_dataset  = FashionDataset(ds_full, test_idx,  label_to_idx, preprocess, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ===== 마지막 레이어 교체 =====
    in_features         = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model               = model.to(device)

    # ===== Class-Balanced Loss =====
    # train set의 클래스별 샘플 수를 세어 weight를 계산하고
    # CrossEntropyLoss의 weight 인자로 전달합니다.
    # 샘플이 적은 클래스일수록 loss 기여도가 높아져 희귀 클래스 학습이 강화됩니다.
    print("===== Computing Class Weights for Balanced Loss =====")
    class_weights = compute_class_weights(train_idx, ds_full, label_to_idx, num_classes, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    print(f"CrossEntropyLoss with class weights applied ({num_classes} classes)\n")

    optimizer = (
        optim.Adam(model.parameters(), lr=args.lr)
        if args.optimizer == "adam"
        else optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    )

    # ===== CSV 로깅 설정 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = f"training_log_{timestamp}.csv"

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "epochs", "lr", "optimizer", "data_dir", "save_model", "model_path",
                         "augmentation", "loss"])
        writer.writerow([args.batch_size, args.epochs, args.lr, args.optimizer, args.data_dir,
                         args.save_model, args.model_path,
                         "HorizontalFlip+Rotation15+ColorJitter(train only)",
                         "CrossEntropyLoss(class_balanced_weight)"])
        writer.writerow([])
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    # ===== 학습 루프 =====
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "final_test_acc": None}

    print("===== Starting Training =====")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch:>2}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        with open(csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc])

    # ===== 최종 테스트 평가 =====
    print("\n===== Final Test Evaluation =====")
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {final_test_loss:.4f}  Final Test Acc: {final_test_acc:.4f}")
    history["final_test_acc"] = final_test_acc

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["final_test_loss", "final_test_acc"])
        writer.writerow([final_test_loss, final_test_acc])

    # ===== 시각화 저장 =====
    save_training_curves(history, timestamp)


    all_labels_arr, all_preds_arr = evaluate_with_predictions(model, test_loader, device)
    save_confusion_matrix(all_labels_arr, all_preds_arr, idx_to_label, timestamp)
    save_per_class_prediction_bars(all_labels_arr, all_preds_arr, idx_to_label, timestamp)

    # ===== 모델 저장 (옵션) =====
    if args.save_model:
        os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.model_path)
        print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()

