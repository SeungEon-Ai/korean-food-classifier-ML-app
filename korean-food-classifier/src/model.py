"""
모델 아키텍처 정의.

새로운 모델 추가하려면:
1. _build_xxx() 함수 작성
2. ARCHITECTURES dict에 등록
3. config.yaml의 model.architecture에서 선택

각 base model마다 preprocess_input이 다름에 주의.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    MobileNetV2,
    EfficientNetB0,
    ResNet50,
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess


def _build_mobilenetv2(image_size: int, num_classes: int, use_pretrained: bool, freeze_base: bool, dropout_rate: float) -> Model:
    """MobileNetV2 기반 모델. 모바일 배포에 가장 추천."""
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = layers.Lambda(mobilenet_preprocess, name="preprocess")(inputs)

    base = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet" if use_pretrained else None,
    )
    base.trainable = not freeze_base

    x = base(x, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="mobilenetv2_classifier")


def _build_efficientnet_b0(image_size: int, num_classes: int, use_pretrained: bool, freeze_base: bool, dropout_rate: float) -> Model:
    """EfficientNet-B0. MobileNetV2보다 정확도 약간 높지만 사이즈도 약간 큼."""
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = layers.Lambda(efficientnet_preprocess, name="preprocess")(inputs)

    base = EfficientNetB0(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet" if use_pretrained else None,
    )
    base.trainable = not freeze_base

    x = base(x, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="efficientnet_b0_classifier")


def _build_resnet50(image_size: int, num_classes: int, use_pretrained: bool, freeze_base: bool, dropout_rate: float) -> Model:
    """ResNet50. 정확도 비교용. 모바일엔 무거움."""
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = layers.Lambda(resnet_preprocess, name="preprocess")(inputs)

    base = ResNet50(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet" if use_pretrained else None,
    )
    base.trainable = not freeze_base

    x = base(x, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="resnet50_classifier")


# 새 아키텍처 추가시 여기에 등록
ARCHITECTURES = {
    "mobilenetv2": _build_mobilenetv2,
    "efficientnet_b0": _build_efficientnet_b0,
    "resnet50": _build_resnet50,
}


def build_model(
    architecture: str,
    image_size: int,
    num_classes: int,
    use_pretrained: bool = True,
    freeze_base: bool = True,
    dropout_rate: float = 0.3,
) -> Model:
    """설정에 맞는 모델을 생성.

    Args:
        architecture: "mobilenetv2" / "efficientnet_b0" / "resnet50"
        image_size: 입력 이미지 사이즈
        num_classes: 클래스 수
        use_pretrained: ImageNet 가중치 사용
        freeze_base: base model freeze (True = transfer learning)
        dropout_rate: classifier head의 dropout

    Returns:
        컴파일 안 된 keras Model
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"지원하지 않는 architecture: {architecture}. "
            f"사용 가능: {list(ARCHITECTURES.keys())}"
        )

    builder = ARCHITECTURES[architecture]
    return builder(image_size, num_classes, use_pretrained, freeze_base, dropout_rate)


if __name__ == "__main__":
    # 단독 실행시 모델 구조 출력
    from src.config import load_config

    config = load_config()
    model = build_model(
        architecture=config["model"]["architecture"],
        image_size=config["data"]["image_size"],
        num_classes=10,  # 예시
        use_pretrained=config["model"]["use_pretrained"],
        freeze_base=config["model"]["freeze_base"],
        dropout_rate=config["model"]["dropout_rate"],
    )
    model.summary()
