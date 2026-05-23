"""
학습된 keras 모델을 TFLite로 변환.

양자화 옵션:
- none: 변환만 (가장 정확, 가장 큼)
- dynamic: 동적 양자화 (사이즈 1/4, 정확도 거의 동일) — 기본 추천
- float16: float16 양자화 (사이즈 1/2, GPU에서 빠름)
- int8: int8 완전 양자화 (사이즈 1/4, 모바일 CPU 최고 성능, 정확도 약간 손실)

사용법:
    python -m src.convert_tflite
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import load_config
from src.data_loader import build_dataset


def representative_dataset_gen(dataset, num_samples: int = 100):
    """int8 양자화에 필요한 representative dataset.

    학습 데이터의 일부를 사용해서 양자화 범위를 결정.
    """
    count = 0
    for images, _ in dataset:
        for img in images:
            if count >= num_samples:
                return
            yield [np.expand_dims(img.numpy().astype(np.float32), axis=0)]
            count += 1


def convert(
    model_path: str,
    output_path: str,
    quantization: str = "dynamic",
    representative_dataset=None,
):
    """keras 모델을 TFLite로 변환.

    Args:
        model_path: 입력 keras 모델 경로
        output_path: 출력 tflite 경로
        quantization: "none" / "dynamic" / "float16" / "int8"
        representative_dataset: int8 양자화시 필요
    """
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "none":
        pass
    elif quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        if representative_dataset is None:
            raise ValueError("int8 양자화엔 representative_dataset 필요")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    else:
        raise ValueError(f"알 수 없는 quantization: {quantization}")

    tflite_model = converter.convert()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    size_mb = len(tflite_model) / 1024 / 1024
    print(f"[완료] TFLite 모델 저장: {output_path}")
    print(f"  사이즈: {size_mb:.2f} MB")
    print(f"  양자화: {quantization}")

    return output_path


def main():
    config = load_config()

    model_path = Path(config["training"]["final_model_path"])
    if not model_path.exists():
        model_path = Path(config["training"]["checkpoint_dir"]) / "best_model.keras"

    if not model_path.exists():
        raise FileNotFoundError("학습된 모델 없음. 먼저 train.py 실행 필요.")

    quantization = config["tflite"]["quantization"]
    output_path = config["tflite"]["output_path"]

    # int8 양자화면 representative dataset 준비
    rep_dataset = None
    if quantization == "int8":
        print("[준비] int8 양자화용 representative dataset 생성")
        processed_dir = Path(config["data"]["processed_dir"])
        train_ds = build_dataset(
            str(processed_dir / "train"),
            image_size=config["data"]["image_size"],
            batch_size=1,
            shuffle=True,
            augment=False,
        )
        rep_dataset = lambda: representative_dataset_gen(train_ds, num_samples=100)

    convert(
        model_path=str(model_path),
        output_path=output_path,
        quantization=quantization,
        representative_dataset=rep_dataset,
    )


if __name__ == "__main__":
    main()
