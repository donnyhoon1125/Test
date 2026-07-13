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
