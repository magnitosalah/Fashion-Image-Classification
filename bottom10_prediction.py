import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random
import os

# ==========================================
# ⚙️ 설정
# ==========================================
MODEL_PATH = "./efficientnet_v2s_final.pth"
NUM_SAMPLES = 8  # 뽑아낼 이미지 개수 (2행 4열 = 8개)

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

    return test_idx # 여기서는 test_idx만 사용

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. 데이터셋 로드 및 하위 10개 클래스 식별
    print("⏳ 데이터셋 로드 및 희귀 클래스 분석 중...")
    ds_full = load_dataset("ashraq/fashion-product-images-small", split="train")
    
    unique_labels = sorted(set(ds_full["articleType"]))
    label_to_idx  = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label  = {idx: label for label, idx in label_to_idx.items()}
    all_labels    = [label_to_idx[lbl] for lbl in ds_full["articleType"]]
    num_classes   = len(unique_labels)

    class_counts = defaultdict(int)
    for lbl in all_labels:
        class_counts[lbl] += 1

    # 빈도수 기준 오름차순 정렬하여 하위 10개 인덱스 추출
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    bottom_10_indices = set([cls_idx for cls_idx, count in sorted_classes[:10]])

    # 2. Test 셋 인덱스 추출 및 하위 10개에 해당하는 것만 필터링
    test_idx = stratified_split(all_labels)
    bottom10_test_idx = [idx for idx in test_idx if all_labels[idx] in bottom_10_indices]

    if len(bottom10_test_idx) == 0:
        print("❌ Test 데이터셋에 하위 10개 클래스의 이미지가 없습니다.")
        return

    # 화면에 보여줄 샘플 무작위 선택
    num_to_sample = min(NUM_SAMPLES, len(bottom10_test_idx))
    sample_indices = random.sample(bottom10_test_idx, num_to_sample)

    # 3. 모델 로드 및 전처리 설정
    print("⏳ 모델을 불러오는 중입니다...")
    weights = EfficientNet_V2_S_Weights.DEFAULT
    preprocess = weights.transforms()

    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 4. 이미지 추론 및 시각화 준비
    print("🎨 시각화 이미지를 생성하는 중입니다...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Sample Predictions for Bottom 10 Rare Classes", fontsize=20, fontweight="bold")
    axes = axes.flatten()

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            sample = ds_full[idx]
            pil_img = sample["image"].convert("RGB")
            true_label_idx = label_to_idx[sample["articleType"]]
            true_label_name = idx_to_label[true_label_idx]

            # 모델 추론
            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            pred_label_name = idx_to_label[predicted_idx.item()]

            # 시각화 플롯팅
            ax = axes[i]
            ax.imshow(pil_img)
            ax.axis("off")

            # 맞추면 초록색, 틀리면 빨간색
            color = "green" if true_label_idx == predicted_idx.item() else "red"
            
            title_text = f"True: {true_label_name}\nPred: {pred_label_name}"
            ax.set_title(title_text, color=color, fontsize=14)

    # 빈 칸 지우기
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # 타이틀과 이미지 간격 조정
    
    # 5. 결과 저장
    save_path = "./bottom10_predictions.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"✅ 완료! 시각화 이미지가 저장되었습니다: {save_path}")

if __name__ == "__main__":
    main()