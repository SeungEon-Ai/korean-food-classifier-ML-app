"""
학습된 모델로 단일 이미지 분류 (테스트용).

사용법:
    python -m src.predict path/to/image.jpg
"""
import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import load_config


def predict_image(image_path: str, model_path: str = None, top_k: int = 3) -> list:
    """이미지 한 장에 대한 분류 결과 반환.

    Returns:
        [(class_name, probability), ...] top_k개
    """
    config = load_config()

    if model_path is None:
        model_path = config["training"]["final_model_path"]
        if not Path(model_path).exists():
            model_path = str(Path(config["training"]["checkpoint_dir"]) / "best_model.keras")

    # 클래스 이름 로드
    class_names_path = Path(config["training"]["checkpoint_dir"]) / "class_names.json"
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # 모델 로드
    model = tf.keras.models.load_model(model_path)

    # 이미지 로드 및 전처리
    image_size = config["data"]["image_size"]
    img = tf.keras.utils.load_img(image_path, target_size=(image_size, image_size))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 예측 (모델 내부에 preprocess_input 포함됨)
    preds = model.predict(img_array, verbose=0)[0]

    # top-k
    top_indices = np.argsort(preds)[::-1][:top_k]
    results = [(class_names[i], float(preds[i])) for i in top_indices]
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python -m src.predict <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    results = predict_image(image_path)

    print(f"\n[예측 결과] {image_path}")
    for i, (cls, prob) in enumerate(results, 1):
        print(f"  {i}. {cls}: {prob * 100:.2f}%")
