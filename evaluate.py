"""테스트셋 평가와 confusion matrix 생성."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import config
import data


def main():
    net = tf.keras.models.load_model(config.MODEL_DIR / "best.keras")
    with open(config.MODEL_DIR / "class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)

    test_ds = data.load("test")
    metrics = net.evaluate(test_ds, return_dict=True)
    print(metrics)

    y_true, y_pred = [], []
    for x, y in test_ds:
        p = net.predict(x, verbose=0)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(p, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.45),) * 2)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    plt.tight_layout()
    plt.savefig(config.MODEL_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
