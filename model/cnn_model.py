"""
model/cnn_model.py

CNN(Convolutional Neural Network) 모델 구조 정의
Fashion-MNIST(28x28 흑백 이미지, 10개 클래스) 분류를 위한
합성곱 신경망.
"""

import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (N, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # (N, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (N, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # (N, 64, 7, 7)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
