"""학습 스크립트.

사용:
    python train.py

split된 데이터가 없으면 data.split_dataset()을 먼저 실행해야 한다.
"""
import json
from pathlib import Path

import tensorflow as tf

import config
import data
import model as model_lib


def main():
    tf.keras.utils.set_random_seed(config.SEED)

    train_ds = data.load("train", augment=True)
    val_ds = data.load("val")

    class_names = sorted(d.name for d in (config.DATA_DIR / "train").iterdir() if d.is_dir())
    num_classes = len(class_names)
    print(f"{num_classes} classes")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    net = model_lib.build(num_classes)
    net.compile(
        optimizer=tf.keras.optimizers.Adam(config.LR),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_DIR / "best.keras",
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6,
        ),
    ]

    history = net.fit(train_ds, validation_data=val_ds,
                      epochs=config.EPOCHS, callbacks=callbacks)

    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(config.MODEL_DIR / "history.json", "w") as f:
        json.dump(history_data, f, indent=2)


if __name__ == "__main__":
    main()
