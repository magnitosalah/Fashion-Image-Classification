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
FINAL_MODEL_PATH = "./efficientnet_v2s.pth"  # 최종 모델 가중치 파일
BATCH_SIZE = 32
MINORITY_THRESHOLD = 50  # 하위 50개 클래스 기준
NUM_SAMPLES_TO_SHOW = 8  # 뽑아낼 이미지 개수 (2x4 그리드용)

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
        real_idx = self.indices[idx]
        sample = self.hf_dataset[real_idx]
        img    = sample["image"].convert("RGB")
        img_tensor = self.preprocess(img)
        label  = self.label_to_idx[sample["articleType"]]
        return img_tensor, label, real_idx

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. 데이터셋 분석 및 소수 클래스 정의
    print("⏳ 데이터셋 분석 및 소수 클래스 필터링 중...")
    ds_full = load_dataset("ashraq/fashion-product-images-small", split="train")
    
    unique_labels = sorted(set(ds_full["articleType"]))
    label_to_idx  = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label  = {idx: label for label, idx in label_to_idx.items()}
    all_labels    = [label_to_idx[lbl] for lbl in ds_full["articleType"]]
    num_classes   = len(unique_labels)

    class_counts = defaultdict(int)
    for lbl in all_labels:
        class_counts[lbl] += 1

    # 하위 50개를 소수 클래스로 정의
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    minority_indices = set([cls_idx for cls_idx, count in sorted_classes[:MINORITY_THRESHOLD]])

    test_idx = stratified_split(all_labels)
    weights = EfficientNet_V2_S_Weights.DEFAULT
    preprocess = weights.transforms()
    test_dataset = FashionTestDataset(ds_full, test_idx, label_to_idx, preprocess)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. 모델 준비
    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    print("🔍 소수 클래스 정답 예측 샘플 수집 중...")
    correct_samples = []

    with torch.no_grad():
        for images, labels, real_indices in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs, 1)

            labels = labels.cpu().numpy()
            predicted = predicted.cpu().numpy()
            confidences = confidences.cpu().numpy()
            real_indices = real_indices.numpy()

            for i in range(len(labels)):
                true_lbl = labels[i]
                pred_lbl = predicted[i]
                
                # 조건: 소수 클래스에 속하고 & 정답을 맞춘 경우
                if true_lbl in minority_indices and true_lbl == pred_lbl:
                    # 중복 클래스가 너무 많이 나오지 않게 조절 (다양성 확보)
                    if not any(s['label_name'] == idx_to_label[true_lbl] for s in correct_samples):
                        correct_samples.append({
                            "real_idx": real_indices[i],
                            "label_name": idx_to_label[true_lbl],
                            "confidence": confidences[i] * 100
                        })
                    
                    if len(correct_samples) >= NUM_SAMPLES_TO_SHOW:
                        break
            if len(correct_samples) >= NUM_SAMPLES_TO_SHOW:
                break

    if len(correct_samples) == 0:
        print("⚠️ 소수 클래스 중 정답을 맞춘 샘플이 없습니다.")
        return

    # 3. 시각화 (원본 이미지 사용)
    print("🎨 정답 샘플 시각화 그리는 중...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, sample in enumerate(correct_samples):
        # 원본 해상도 이미지를 로드하여 예쁘게 시각화
        original_img = ds_full[int(sample["real_idx"])]["image"]
        ax = axes[i]
        ax.imshow(original_img)
        ax.axis('off')
        
        label_name = sample["label_name"]
        if len(label_name) > 15:
            label_name = label_name[:12] + "..."
            
        title_text = f"True: {label_name}\nPred: {label_name}\nConf: {sample['confidence']:.1f}%"
        # 정답이니까 깔끔한 초록색 텍스트 사용
        ax.set_title(title_text, color='#2E7D32', fontsize=13, fontweight='bold', pad=10)

    # 남는 빈 칸 숨기기
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_path = "./correct_predictions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 완료! 시각화 이미지가 저장되었습니다: {save_path}")

if __name__ == "__main__":
    main()