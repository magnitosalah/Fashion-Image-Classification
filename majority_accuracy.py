import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ 설정 (본인의 파일명에 맞게 수정하세요)
# ==========================================
BASELINE_MODEL_PATH = "./baseline.pth"       # 베이스라인 가중치 파일
FINAL_MODEL_PATH = "./efficientnet_v2s.pth"  # 최종 모델 가중치 파일
BATCH_SIZE = 32

def stratified_split(labels, train_ratio=0.6, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    test_idx = []
    for label, indices in class_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n       = len(indices)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        test_idx.extend(indices[n_train + n_val:].tolist())

    return test_idx

class FashionTestDataset(Dataset):
    def __init__(self, hf_dataset, indices, label_to_idx, preprocess):
        self.hf_dataset   = hf_dataset
        self.indices      = indices
        self.label_to_idx = label_to_idx
        self.preprocess   = preprocess

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.hf_dataset[self.indices[idx]]
        img    = sample["image"].convert("RGB")
        img    = self.preprocess(img)
        label  = self.label_to_idx[sample["articleType"]]
        return img, label

def evaluate_model(model_path, model_base, test_loader, target_indices, device):
    model_base.load_state_dict(torch.load(model_path, map_location=device))
    model_base = model_base.to(device)
    model_base.eval()
    
    results = {cls_idx: {"correct": 0, "total": 0} for cls_idx in target_indices}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_base(images)
            _, predicted = torch.max(outputs, 1)

            labels = labels.cpu().numpy()
            predicted = predicted.cpu().numpy()

            for true_lbl, pred_lbl in zip(labels, predicted):
                if true_lbl in target_indices:
                    results[true_lbl]["total"] += 1
                    if true_lbl == pred_lbl:
                        results[true_lbl]["correct"] += 1
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. 데이터셋 분석 및 상위 10개 다수 클래스 정의
    print("⏳ 데이터셋 분석 및 Test Dataloader 구성 중...")
    ds_full = load_dataset("ashraq/fashion-product-images-small", split="train")
    
    unique_labels = sorted(set(ds_full["articleType"]))
    label_to_idx  = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label  = {idx: label for label, idx in label_to_idx.items()}
    all_labels    = [label_to_idx[lbl] for lbl in ds_full["articleType"]]
    num_classes   = len(unique_labels)

    class_counts = defaultdict(int)
    for lbl in all_labels:
        class_counts[lbl] += 1

    # 빈도수 기준 '내림차순' 정렬하여 상위 10개 추출 (Majority Classes)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    top_10_majority = sorted_classes[:10]
    top_10_indices = set([cls_idx for cls_idx, count in top_10_majority])

    test_idx = stratified_split(all_labels)
    weights = EfficientNet_V2_S_Weights.DEFAULT
    preprocess = weights.transforms()
    test_dataset = FashionTestDataset(ds_full, test_idx, label_to_idx, preprocess)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. 모델 준비 및 평가
    model_base = efficientnet_v2_s(weights=None)
    in_features = model_base.classifier[1].in_features
    model_base.classifier[1] = nn.Linear(in_features, num_classes)

    print("🔍 베이스라인 모델 평가 중...")
    baseline_results = evaluate_model(BASELINE_MODEL_PATH, model_base, test_loader, top_10_indices, device)
    
    print("🔍 최종 모델 평가 중...")
    final_results = evaluate_model(FINAL_MODEL_PATH, model_base, test_loader, top_10_indices, device)

    # 3. 데이터 정리
    class_names = []
    baseline_accs = []
    final_accs = []

    for cls_idx, count in top_10_majority:
        name = idx_to_label[cls_idx]
        if len(name) > 10: 
            name = name.replace(" ", "\n", 1)
        class_names.append(name)
        
        b_res = baseline_results[cls_idx]
        b_acc = (b_res["correct"] / b_res["total"] * 100) if b_res["total"] > 0 else 0
        baseline_accs.append(b_acc)
        
        f_res = final_results[cls_idx]
        f_acc = (f_res["correct"] / f_res["total"] * 100) if f_res["total"] > 0 else 0
        final_accs.append(f_acc)

    # 4. 시각화 (Grouped Bar Chart)
    print("🎨 다수 클래스 비교 그래프 그리는 중...")
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    # 다수 클래스는 기존 소수 클래스 그래프와 색상을 맞춰 통일감을 줍니다.
    rects1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline Model', color='#B0BEC5', edgecolor='black')
    rects2 = ax.bar(x + width/2, final_accs, width, label='Final Model (Class-Balanced)', color='#4A90E2', edgecolor='black')

    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Top 10 Majority Classes Accuracy: Baseline vs Final\n(Trade-off Analysis)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    
    # 다수 클래스는 정확도가 높을 테니 y축 범위를 80~105 등으로 좁히면 차이가 더 잘 보일 수 있습니다.
    # 하지만 전체적인 통일성을 위해 0~105로 설정합니다.
    ax.set_ylim(0, 110) 
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    
    save_path = "./majority_classes_accuracy.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 완료! 시각화 그래프가 저장되었습니다: {save_path}")

if __name__ == "__main__":
    main()