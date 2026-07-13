"""
data/data_load.py

Fashion-MNIST 데이터셋을 다운로드하고 로드하는 코드
- 학습(train) / 테스트(test) 데이터 분리
- 데이터 정규화 및 Tensor 변환 처리
- DataLoader 생성 (batch 단위 학습 지원)
"""

import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Fashion-MNIST 원본 파일은 torchvision 기본 미러(AWS S3) 대신
# GitHub 공식 저장소의 raw 파일을 이용해 다운로드한다.
BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

DATA_DIR = Path(__file__).resolve().parent / "raw"

# ImageNet 스타일이 아닌, Fashion-MNIST 자체 통계 기반 정규화 값
MNIST_MEAN = 0.2860
MNIST_STD = 0.3530


def _download_file(filename):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / filename

    if dest.exists():
        return dest

    url = BASE_URL + filename
    print(f"[다운로드] {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[완료] {dest} ({dest.stat().st_size / 1024:.1f} KB)")
    return dest


def _load_images(gz_path):
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    # IDX3 헤더: magic(4) + num_images(4) + rows(4) + cols(4)
    num_images = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(num_images, rows, cols)
    return images


def _load_labels(gz_path):
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    # IDX1 헤더: magic(4) + num_labels(4)
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels


def download_fashion_mnist():
    """4개의 원본 파일을 다운로드하고 numpy 배열로 반환한다."""
    paths = {key: _download_file(name) for key, name in FILES.items()}

    train_images = _load_images(paths["train_images"])
    train_labels = _load_labels(paths["train_labels"])
    test_images = _load_images(paths["test_images"])
    test_labels = _load_labels(paths["test_labels"])

    print(f"학습 이미지 수: {len(train_images)}")
    print(f"테스트 이미지 수: {len(test_images)}")
    print(f"클래스: {CLASS_NAMES}")

    return train_images, train_labels, test_images, test_labels


class FashionMNISTDataset(Dataset):
    """정규화 및 Tensor 변환이 적용된 Fashion-MNIST Dataset."""

    def __init__(self, images, labels):
        # (N, 28, 28) uint8 -> (N, 1, 28, 28) float32, [0, 1] 범위로 스케일링
        images = images.astype(np.float32) / 255.0
        images = (images - MNIST_MEAN) / MNIST_STD
        self.images = torch.from_numpy(images).unsqueeze(1)  # (N, 1, 28, 28)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def build_loaders(batch_size=64, num_workers=0):
    """학습/테스트 DataLoader를 생성해서 반환한다."""
    train_images, train_labels, test_images, test_labels = download_fashion_mnist()

    train_dataset = FashionMNISTDataset(train_images, train_labels)
    test_dataset = FashionMNISTDataset(test_images, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataset, test_dataset, train_loader, test_loader


if __name__ == "__main__":
    train_dataset, test_dataset, train_loader, test_loader = build_loaders()
    images, labels = next(iter(train_loader))
    print(f"배치 이미지 shape: {images.shape}")
    print(f"배치 라벨 shape: {labels.shape}")
