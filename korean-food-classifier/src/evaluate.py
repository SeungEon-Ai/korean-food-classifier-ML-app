"""
학습된 모델 평가 스크립트.

사용법:
    python -m src.evaluate

출력:
- 테스트셋 정확도
- 클래스별 정확도
- Confusion matrix 이미지 (models/ 폴더에 저장)
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from src.config import load_config
from src.data_loader import build_dataset


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str):
    """Confusion matrix를 이미지로 저장."""
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.7),) * 2)

    # 정규화
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")

    # 셀에 숫자 표시
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm_norm[i, j]:.2f}"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[완료] Confusion matrix 저장: {output_path}")


def main():
    config = load_config()

    # 모델 로드
    model_path = Path(config["training"]["final_model_path"])
    if not model_path.exists():
        # final이 없으면 best checkpoint 시도
        model_path = Path(config["training"]["checkpoint_dir"]) / "best_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"학습된 모델 없음. 먼저 train.py 실행 필요.")

    print(f"[로드] {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 클래스 이름 로드
    class_names_path = Path(config["training"]["checkpoint_dir"]) / "class_names.json"
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # 테스트 데이터셋
    processed_dir = Path(config["data"]["processed_dir"])
    test_ds = build_dataset(
        str(processed_dir / "test"),
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        augment=False,
    )

    # 평가
    print("\n=== 전체 평가 ===")
    results = model.evaluate(test_ds, verbose=1, return_dict=True)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # 예측 수집 (confusion matrix용)
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n=== 클래스별 리포트 ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    output_path = model_path.parent / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, str(output_path))


if __name__ == "__main__":
    main()
