"""
training/train.py

모델 학습을 수행하는 메인 코드
- loss 함수 및 optimizer 설정
- epoch 단위 학습 진행
- 학습 과정에서 loss 및 accuracy 출력
- MLP와 CNN 모델을 각각 적용하여 학습(train)을 수행하고
  검증(validation, 여기서는 test set 기준) 평가를 진행한다.

[구조 변경 사항]
이 파일은 이제 training/ 폴더에 있고, 결과 저장 위치는 output/ 폴더로
분리되었다. 따라서 저장 경로를 '내 폴더(__file__ 기준)'가 아니라
'프로젝트 루트 기준 output/ 폴더'로 명시적으로 지정한다.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# training/train.py 기준 프로젝트 루트 = 한 단계 위
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data.data_load import build_loaders  # noqa: E402
from model.mlp_model import MLPClassifier  # noqa: E402
from model.cnn_model import CNNClassifier  # noqa: E402

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# [변경] 더 이상 Path(__file__).resolve().parent 가 아니라
# 프로젝트 루트 아래의 output/ 폴더를 직접 지정한다.
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # output/ 폴더가 없으면 생성


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total


def train_model(model_name, model, train_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_val_acc = 0.0
    ckpt_path = OUTPUT_DIR / f"{model_name}_best.pth"

    print(f"\n==== {model_name.upper()} 학습 시작 ====")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{model_name.upper()}] Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_accuracy": best_val_acc,
            }, ckpt_path)

    print(f"{model_name.upper()} 최고 검증 정확도: {best_val_acc:.2f}% (저장: output/{ckpt_path.name})")

    return history, best_val_acc


def main():
    device = torch.device("cpu")
    print(f"사용 장치: {device}")

    train_dataset, test_dataset, train_loader, test_loader = build_loaders(batch_size=BATCH_SIZE)

    all_history = {}

    mlp_model = MLPClassifier().to(device)
    mlp_history, mlp_best = train_model("mlp", mlp_model, train_loader, test_loader, device)
    all_history["mlp"] = mlp_history

    cnn_model = CNNClassifier().to(device)
    cnn_history, cnn_best = train_model("cnn", cnn_model, train_loader, test_loader, device)
    all_history["cnn"] = cnn_history

    history_path = OUTPUT_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(all_history, f, ensure_ascii=False, indent=2)

    print(f"\n학습 기록 저장 완료: output/{history_path.name}")
    print("\n==== 최종 비교 ====")
    print(f"MLP 최고 검증 정확도: {mlp_best:.2f}%")
    print(f"CNN 최고 검증 정확도: {cnn_best:.2f}%")


if __name__ == "__main__":
    main()
