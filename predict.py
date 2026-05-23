"""단일 이미지를 분류해서 top-5 결과를 출력.

사용:
    python predict.py path/to/image.jpg
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

import config


def predict(image_path, model_path=None, top=5):
    model_path = model_path or config.MODEL_DIR / "best.keras"
    with open(config.MODEL_DIR / "class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)

    net = tf.keras.models.load_model(model_path)

    img = tf.keras.utils.load_img(image_path, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))
    x = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
    probs = net.predict(x, verbose=0)[0]

    idx = np.argsort(probs)[::-1][:top]
    return [(class_names[i], float(probs[i])) for i in idx]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python predict.py <image>")
        sys.exit(1)

    for name, prob in predict(sys.argv[1]):
        print(f"{name:15s} {prob * 100:5.1f}%")
