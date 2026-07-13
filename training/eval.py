"""
training/eval.py

학습된 모델(MLP, CNN)을 이용한 테스트 데이터 평가
- 전체 테스트 데이터에 대한 정확도 계산
- 예측 결과와 정답 비교 (클래스별 정확도 포함)
- MLP와 CNN의 분류 성능 차이를 비교

[구조 변경 사항]
train.py가 저장한 가중치(.pth)는 이제 output/ 폴더에 있으므로,
그 경로를 프로젝트 루트 기준으로 다시 찾아가도록 수정했다.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data.data_load import build_loaders, CLASS_NAMES  # noqa: E402
from model.mlp_model import MLPClassifier  # noqa: E402
from model.cnn_model import CNNClassifier  # noqa: E402

# [변경] eval.py는 training/ 폴더에 있지만, 읽어와야 할 .pth 파일은
# output/ 폴더에 있으므로 별도로 경로를 지정한다.
OUTPUT_DIR = PROJECT_ROOT / "output"


def load_model(model_name, model_class, device):
    ckpt_path = OUTPUT_DIR / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"output/{ckpt_path.name} 파일이 없습니다. 먼저 training/train.py를 실행하세요."
        )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = model_class().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("best_val_accuracy")


def evaluate_model(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for label, prediction in zip(labels, predictions):
                label = int(label.item())
                prediction = int(prediction.item())
                class_total[label] = class_total.get(label, 0) + 1
                if label == prediction:
                    class_correct[label] = class_correct.get(label, 0) + 1

    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    return avg_loss, accuracy, correct, total, class_correct, class_total


def print_report(model_name, avg_loss, accuracy, correct, total, class_correct, class_total):
    print(f"\n==== {model_name.upper()} 평가 결과 ====")
    print(f"테스트 이미지 수: {total}")
    print(f"맞힌 개수: {correct}")
    print(f"평균 손실값: {avg_loss:.4f}")
    print(f"전체 정확도: {accuracy:.2f}%")

    print(f"\n[{model_name.upper()}] 클래스별 정확도")
    for class_index, class_name in enumerate(CLASS_NAMES):
        c_total = class_total.get(class_index, 0)
        c_correct = class_correct.get(class_index, 0)
        if c_total == 0:
            print(f"  {class_name}: 이미지 없음")
        else:
            print(f"  {class_name}: {c_correct}/{c_total} = {100 * c_correct / c_total:.2f}%")


def main():
    device = torch.device("cpu")

    _, _, _, test_loader = build_loaders()

    results = {}

    for model_name, model_class in [("mlp", MLPClassifier), ("cnn", CNNClassifier)]:
        model, best_val_acc = load_model(model_name, model_class, device)
        avg_loss, accuracy, correct, total, class_correct, class_total = evaluate_model(
            model, test_loader, device
        )
        print_report(model_name, avg_loss, accuracy, correct, total, class_correct, class_total)
        results[model_name] = {"accuracy": accuracy, "avg_loss": avg_loss}

    print("\n==== MLP vs CNN 최종 비교 ====")
    print(f"MLP 전체 정확도: {results['mlp']['accuracy']:.2f}%")
    print(f"CNN 전체 정확도: {results['cnn']['accuracy']:.2f}%")
    diff = results["cnn"]["accuracy"] - results["mlp"]["accuracy"]
    better = "CNN" if diff > 0 else "MLP"
    print(f"{better} 모델이 {abs(diff):.2f}%p 더 높은 정확도를 보였습니다.")

    print("\n평가 완료")


if __name__ == "__main__":
    main()
