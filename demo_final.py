import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights
from datasets import load_dataset
import gradio as gr
from PIL import Image

# ==========================================
# 1. 모델 및 데이터셋 초기 설정 (기존과 동일)
# ==========================================
MODEL_PATH = "./efficientnet_v2s.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 라벨 딕셔너리 로드 (시간 단축을 위해 하드코딩해도 되지만, 원본 유지)
print("라벨 정보를 불러오는 중...")
ds_full = load_dataset("ashraq/fashion-product-images-small", split="train")
unique_labels = sorted(set(ds_full["articleType"]))
idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
num_classes = len(unique_labels)

# 모델 구조 세팅 및 가중치 로드
weights = EfficientNet_V2_S_Weights.DEFAULT
preprocess = weights.transforms()

model = efficientnet_v2_s(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ==========================================
# 2. 예측 함수 정의 (Gradio가 호출할 함수)
# ==========================================
def predict_image(img):
    # img는 Gradio에서 전달받은 PIL Image 객체입니다.
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Gradio의 Label 컴포넌트가 읽을 수 있도록 {클래스명: 확률} 딕셔너리 형태로 반환
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result_dict = {idx_to_label[top5_catid[i].item()]: top5_prob[i].item() for i in range(5)}
    
    return result_dict

# ==========================================
# 3. Gradio 웹 인터페이스 세팅
# ==========================================
# 입력란: 이미지 드래그 앤 드롭 가능 (type="pil"로 설정하여 PIL 이미지로 받음)
image_input = gr.Image(type="pil", label="옷 사진을 드래그 앤 드롭 하세요!")

# 출력란: 상위 예측 결과와 확률 바 그래프
label_output = gr.Label(num_top_classes=5, label="인공지능 분석 결과")

# 웹 화면 구성
demo = gr.Interface(
    fn=predict_image,               # 실행할 예측 함수
    inputs=image_input,             # 웹의 입력 창
    outputs=label_output,           # 웹의 결과 창
    title="👕 AI 패션 아이템 분류기_final",
    description="사진을 올리면 이 옷이 어떤 종류인지 인공지능이 분석해 줍니다."
)

if __name__ == "__main__":
    # share=True로 설정하면, 외부에서도 접속할 수 있는 임시 공개 링크가 72시간 동안 생성됩니다.
    demo.launch(share=True)