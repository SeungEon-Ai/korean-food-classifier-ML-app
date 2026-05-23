"""
학습 스크립트.

사용법:
    python -m src.train

학습 도중 ctrl+C 로 중단 가능. checkpoint는 자동 저장됨.
"""
import json
from pathlib import Path

import tensorflow as tf

from src.config import load_config
from src.data_loader import prepare_dataset, build_dataset
from src.model import build_model


def main():
    config = load_config()

    # 재현 가능성을 위한 시드 설정
    seed = config["misc"]["random_seed"]
    tf.keras.utils.set_random_seed(seed)

    # 1. 데이터 준비
    print("\n" + "=" * 60)
    print("1단계: 데이터 준비")
    print("=" * 60)
    info = prepare_dataset(
        raw_dir=config["data"]["raw_dir"],
        processed_dir=config["data"]["processed_dir"],
        target_classes=config["data"]["target_classes"] or None,
        max_images_per_class=config["data"]["max_images_per_class"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=seed,
    )

    processed_dir = Path(config["data"]["processed_dir"])
    image_size = config["data"]["image_size"]
    batch_size = config["training"]["batch_size"]

    train_ds = build_dataset(
        str(processed_dir / "train"),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        augment=config["training"]["use_augmentation"],
    )
    val_ds = build_dataset(
        str(processed_dir / "val"),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )

    # 클래스 이름 가져오기 (tf.keras가 자동으로 정렬된 순서로 라벨링)
    class_names = sorted([d.name for d in (processed_dir / "train").iterdir() if d.is_dir()])
    num_classes = len(class_names)
    print(f"\n클래스 수: {num_classes}")
    print(f"클래스: {class_names}")

    # 클래스 이름 저장 (추론할 때 필요)
    Path(config["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(config["training"]["checkpoint_dir"]) / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # 2. 모델 빌드
    print("\n" + "=" * 60)
    print("2단계: 모델 빌드")
    print("=" * 60)
    model = build_model(
        architecture=config["model"]["architecture"],
        image_size=image_size,
        num_classes=num_classes,
        use_pretrained=config["model"]["use_pretrained"],
        freeze_base=config["model"]["freeze_base"],
        dropout_rate=config["model"]["dropout_rate"],
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    model.summary()

    # 3. 콜백 설정
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(config["training"]["checkpoint_dir"]) / "best_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config["training"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 4. 학습
    print("\n" + "=" * 60)
    print("3단계: 학습 시작")
    print("=" * 60)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # 5. 최종 모델 저장
    final_path = Path(config["training"]["final_model_path"])
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_path)
    print(f"\n[완료] 최종 모델 저장: {final_path}")

    # 학습 이력 저장
    history_path = final_path.parent / "training_history.json"
    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"[완료] 학습 이력 저장: {history_path}")


if __name__ == "__main__":
    main()
