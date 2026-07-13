# 👕 Fashion MNIST 분류 프로젝트

Fashion MNIST 데이터를 활용하여 다층 퍼셉트론(MLP) 및 합성곱 신경망(CNN) 모델을 학습하고 평가하는 프로젝트입니다.

---

## 📂 프로젝트 구조 (Directory Structure)

```text
my-fashion-mnist-project/
│
├── data/                          # 1. 데이터 관련 폴더
│   └── download_data.py           # - 오픈소스 데이터를 다운로드하는 스크립트
│
├── models/                        # 2. 모델 정의 관련 폴더
│   ├── mlp.py                     # - MLP(Multi-Layer Perceptron) 모델 정의
│   └── cnn.py                     # - CNN(Convolutional Neural Network) 모델 정의
│
├── training/                           # 3. 실행 및 학습 관련 폴더
│   ├── train.py                   # - 모델 학습(Training) 스크립트
│   ├── evaluate.py                # - 모델 평가(Evaluation) 스크립트
│   └── visualize.py               # - 결과 시각화(Visualization) 스크립트
│
└── output/                        # 4. 결과물 저장 폴더
    ├── loss_plot.png              # - 시각화된 손실(Loss) 그래프 결과
    ├── model_weights.pth          # - 학습이 완료된 모델 가중치 파일
    └── result.json                # - 각 단계별 학습 결과 및 평가지표 기록 (JSON)
## 🚀 실행 및 실험 방법

### 1. 전체 모델 비교 학습 (기본)
MLP 모델과 CNN 모델을 연속으로 학습시키고 최종 성능을 비교하려면 아래 명령어를 실행합니다.
```bash
python training/train.py
```
def main():
    # ... (생략) ...
    
    # 1. MLP만 테스트하고 싶을 때: CNN 호출 부분을 주석 처리(#)합니다.
    mlp_model = MLPClassifier().to(device)
    mlp_history, mlp_best = train_model("mlp", mlp_model, train_loader, test_loader, device)
    all_history["mlp"] = mlp_history

    # 2. CNN만 테스트하고 싶을 때: MLP 호출 부분을 주석 처리(#)합니다.
    # cnn_model = CNNClassifier().to(device)
    # cnn_history, cnn_best = train_model("cnn", cnn_model, train_loader, test_loader, device)
