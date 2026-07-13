"""
model/mlp_model.py

MLP(Multi-Layer Perceptron) 모델 구조 정의
Fashion-MNIST(28x28 흑백 이미지, 10개 클래스) 분류를 위한
완전연결층 기반 신경망.
"""

import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),                  # (N, 1, 28, 28) -> (N, 784)
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)
