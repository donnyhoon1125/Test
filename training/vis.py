"""
training/vis.py

학습 결과 시각화 코드
- loss 및 accuracy 변화 그래프 출력 (MLP vs CNN 비교)
- 샘플 이미지에 대한 예측 결과 시각화

[구조 변경 사항]
training_history.json, .pth 가중치, 그리고 최종 그래프 이미지까지
모두 output/ 폴더를 기준으로 읽고 쓰도록 경로를 수정했다.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data.data_load import build_loaders, CLASS_NAMES, MNIST_MEAN, MNIST_STD  # noqa: E402
from model.mlp_model import MLPClassifier  # noqa: E402
from model.cnn_model import CNNClassifier  # noqa: E402

# [변경] vis.py도 training/ 폴더에 있으므로, 읽기(가중치, history)와
# 쓰기(그래프 이미지) 모두 output/ 폴더를 명시적으로 가리키게 한다.
OUTPUT_DIR = PROJECT_ROOT / "output"


def load_model(model_name, model_class, device):
    ckpt_path = OUTPUT_DIR / f"{model_name}_best.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = model_class().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def plot_training_curves(history):
    mlp_hist = history["mlp"]
    cnn_hist = history["cnn"]
    epochs = range(1, len(mlp_hist["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, mlp_hist["val_loss"], label="MLP Val Loss", color="tab:blue")
    plt.plot(epochs, cnn_hist["val_loss"], label="CNN Val Loss", color="tab:orange")
    plt.plot(epochs, mlp_hist["train_loss"], label="MLP Train Loss", color="tab:blue", linestyle="--", alpha=0.5)
    plt.plot(epochs, cnn_hist["train_loss"], label="CNN Train Loss", color="tab:orange", linestyle="--", alpha=0.5)
    plt.title("Loss (MLP vs CNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, mlp_hist["val_acc"], label="MLP Val Accuracy", color="tab:blue")
    plt.plot(epochs, cnn_hist["val_acc"], label="CNN Val Accuracy", color="tab:orange")
    plt.plot(epochs, mlp_hist["train_acc"], label="MLP Train Accuracy", color="tab:blue", linestyle="--", alpha=0.5)
    plt.plot(epochs, cnn_hist["train_acc"], label="CNN Train Accuracy", color="tab:orange", linestyle="--", alpha=0.5)
    plt.title("Accuracy (MLP vs CNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    out_path = OUTPUT_DIR / "training_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"학습 곡선 비교 그래프 저장 완료: output/{out_path.name}")


def denormalize(image_tensor):
    image = image_tensor.numpy().squeeze(0)  # (28, 28)
    image = image * MNIST_STD + MNIST_MEAN
    image = np.clip(image, 0, 1)
    return image


def plot_prediction_grid(mlp_model, cnn_model, test_dataset, device, num_images=8):
    cols = 4
    rows = int(np.ceil(num_images / cols))

    plt.figure(figsize=(4 * cols, 5 * rows))

    for i in range(num_images):
        image_tensor, true_label = test_dataset[i]
        input_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            mlp_pred = mlp_model(input_tensor).argmax(dim=1).item()
            cnn_pred = cnn_model(input_tensor).argmax(dim=1).item()

        true_name = CLASS_NAMES[int(true_label)]
        mlp_name = CLASS_NAMES[mlp_pred]
        cnn_name = CLASS_NAMES[cnn_pred]

        mlp_mark = "OK" if mlp_pred == int(true_label) else "WRONG"
        cnn_mark = "OK" if cnn_pred == int(true_label) else "WRONG"

        plt.subplot(rows, cols, i + 1)
        plt.imshow(denormalize(image_tensor), cmap="gray")
        plt.title(
            f"True: {true_name}\nMLP: {mlp_name} ({mlp_mark})\nCNN: {cnn_name} ({cnn_mark})",
            fontsize=9,
        )
        plt.axis("off")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "prediction_grid.png"
    plt.savefig(out_path)
    plt.close()
    print(f"예측 결과 시각화 저장 완료: output/{out_path.name}")


def main():
    device = torch.device("cpu")

    history_path = OUTPUT_DIR / "training_history.json"
    if not history_path.exists():
        raise FileNotFoundError("output/training_history.json이 없습니다. 먼저 training/train.py를 실행하세요.")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    plot_training_curves(history)

    _, test_dataset, _, _ = build_loaders()

    mlp_model = load_model("mlp", MLPClassifier, device)
    cnn_model = load_model("cnn", CNNClassifier, device)

    plot_prediction_grid(mlp_model, cnn_model, test_dataset, device)

    print("\n시각화 완료")


if __name__ == "__main__":
    main()
