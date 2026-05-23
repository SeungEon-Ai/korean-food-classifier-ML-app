"""학습된 모델을 TFLite로 변환.

사용:
    python export.py                 # dynamic 양자화 (기본)
    python export.py --quant float16
    python export.py --quant none
"""
import argparse

import tensorflow as tf

import config


def convert(model_path, out_path, quant="dynamic"):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quant == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quant != "none":
        raise ValueError(f"unknown quant: {quant}")

    tflite = converter.convert()
    out_path.write_bytes(tflite)
    print(f"{out_path} ({len(tflite) / 1024 / 1024:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="dynamic", choices=["none", "dynamic", "float16"])
    ap.add_argument("--src", default=str(config.MODEL_DIR / "best.keras"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or config.MODEL_DIR / f"kfood_{args.quant}.tflite"
    convert(args.src, out, args.quant)


if __name__ == "__main__":
    main()
