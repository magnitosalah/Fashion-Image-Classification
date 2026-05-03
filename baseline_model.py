import argparse
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_v2_s
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
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
    parser.add_argument("--model_path", type=str, default="./baseline.pth")
    return parser.parse_args()




def stratified_split(dataset, labels, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    class별 비율을 유지하면서 train / val / test 인덱스를 분리합니다.
    train:val:test = 6:2:2
    """
    rng = np.random.default_rng(seed)
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)


    train_idx, val_idx, test_idx = [], [], []


    for label, indices in class_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)


        train_idx.extend(indices[:n_train].tolist())
        val_idx.extend(indices[n_train:n_train + n_val].tolist())
        test_idx.extend(indices[n_train + n_val:].tolist())


    return train_idx, val_idx, test_idx




def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0


    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)


    return total_loss / total, correct / total




def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0


    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)


            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)


    return total_loss / total, correct / total




def evaluate_with_predictions(model, loader, device):
    """테스트셋 전체 예측값과 실제 레이블을 반환합니다 (confusion matrix용)."""
    model.eval()
    all_preds, all_labels = [], []


    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())


    return np.array(all_labels), np.array(all_preds)




def save_training_curves(history, timestamp):
    """train/val loss & accuracy 커브 + final test accuracy 를 이미지로 저장합니다."""
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)


    epochs = range(1, len(history["train_loss"]) + 1)
    final_test_acc = history["final_test_acc"]


    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Training Curves — {timestamp}", fontsize=16, fontweight="bold")


    # Loss 커브
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)


    # Accuracy 커브
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Accuracy")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], "r-o", label="Val Accuracy")
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
    """클래스별 confusion matrix를 이미지로 저장합니다."""
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)


    class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    num_classes = len(class_names)
    cm = confusion_matrix(all_labels, all_preds, labels=sorted(idx_to_label.keys()))


    # 클래스 수에 따라 동적으로 figure 크기 조정 (최소 20, 클래스당 0.35 비율 추가)
    fig_size = max(20, int(num_classes * 0.35))
    font_size = max(5, int(10 - num_classes * 0.03))


    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.3,
        linecolor="gray",
        annot_kws={"size": font_size},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {timestamp}", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=font_size)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=font_size)


    plt.tight_layout()
    path = f"{plot_dir}/confusion_matrix_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to: {path}")




def main():
    args = get_args()


    print("===== Input Arguments =====")
    for k, v in vars(args).items():
        print(f"{k:15s}: {v}")
    print("===========================\n")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # ===== 모델 및 전처리 설정 =====
    weights = None
    model = efficientnet_v2_s(weights=weights)
    preprocess = transforms.Compose([
        transforms.Resize(384), # EfficientNet_V2_S의 기본 입력 크기
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # ===== 데이터셋 로드 =====
    ds_full = load_dataset("ashraq/fashion-product-images-small", split="train")


    unique_labels = sorted(set(ds_full["articleType"]))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    num_classes = len(unique_labels)


    all_labels = [label_to_idx[lbl] for lbl in ds_full["articleType"]]


    # ===== Stratified Split (6:2:2) =====
    train_idx, val_idx, test_idx = stratified_split(ds_full, all_labels, train_ratio=0.6, val_ratio=0.2, seed=42)
    print(f"Dataset Split => Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")


    def transform_fn(examples):
        images = [preprocess(img.convert("RGB")) for img in examples["image"]]
        labels = [label_to_idx[lbl] for lbl in examples["articleType"]]
        return {"image": images, "label": labels}


    ds_full.set_transform(transform_fn)


    def collate_fn(batch):
        images = torch.stack([x["image"] for x in batch])
        labels = torch.tensor([x["label"] for x in batch])
        return images, labels


    train_loader = DataLoader(Subset(ds_full, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(Subset(ds_full, val_idx),   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(Subset(ds_full, test_idx),  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)


    # ===== 마지막 레이어 교체 =====
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.Adam(model.parameters(), lr=args.lr)
        if args.optimizer == "adam"
        else optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    )


    # ===== CSV 로깅 설정 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"training_log_{timestamp}.csv"


    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "epochs", "lr", "optimizer", "data_dir", "save_model", "model_path"])
        writer.writerow([args.batch_size, args.epochs, args.lr, args.optimizer, args.data_dir, args.save_model, args.model_path])
        writer.writerow([])
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])


    # ===== 학습 루프 =====
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "final_test_acc": None}


    print("\n===== Starting Training =====")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)


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


    # final test acc를 CSV에 추가 기록
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["final_test_loss", "final_test_acc"])
        writer.writerow([final_test_loss, final_test_acc])


    # ===== 시각화 저장 =====
    save_training_curves(history, timestamp)


    all_labels_arr, all_preds_arr = evaluate_with_predictions(model, test_loader, device)
    save_confusion_matrix(all_labels_arr, all_preds_arr, idx_to_label, timestamp)


    # ===== 모델 저장 (옵션) =====
    if args.save_model:
        os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.model_path)
        print(f"\nModel saved to: {args.model_path}")




if __name__ == "__main__":
    main()



