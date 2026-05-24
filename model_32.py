"""MobileNetV2 기반 분류기.

preprocess_input은 모델 안에 넣지 않는다. (Lambda 직렬화 문제 회피)
대신 data load 후 한 번만 통과시키거나, 추론 시 직접 호출한다.
"""
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import config


def build(num_classes, freeze_base=True):
    inputs = layers.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)  # preprocess_input과 동치

    base = MobileNetV2(
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = not freeze_base

    x = base(x, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(config.DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="kfood")


# 외부에서 추론할 때 쓸 수 있게 노출
__all__ = ["build", "preprocess_input"]
