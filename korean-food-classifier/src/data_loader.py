"""
데이터 로딩 및 전처리 모듈.

가장 자주 수정될 곳:
- prepare_dataset(): 데이터셋 폴더 구조가 다르면 여기 수정
- get_augmentation(): 데이터 증강 방식 변경

기본 가정:
    raw_dir/
        클래스1/
            *.jpg
        클래스2/
            *.jpg

만약 AI Hub 데이터가 라벨 JSON으로 따로 있다면
prepare_dataset() 함수를 그에 맞게 수정 필요.
"""
from pathlib import Path
from typing import Optional
import random
import shutil

import tensorflow as tf
from tensorflow.keras import layers


def prepare_dataset(
    raw_dir: str,
    processed_dir: str,
    target_classes: Optional[list] = None,
    max_images_per_class: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """원본 데이터를 train/val/test로 분리해서 processed_dir에 복사.

    이미 분리되어 있으면 스킵.

    Args:
        raw_dir: 원본 데이터 경로 (클래스별 폴더 구조)
        processed_dir: 출력 경로
        target_classes: 사용할 클래스 목록 (None이면 전체)
        max_images_per_class: 클래스당 최대 이미지 (빠른 실험용)
        train_ratio, val_ratio, test_ratio: 분할 비율
        seed: 랜덤 시드

    Returns:
        {"classes": [...], "num_train": N, ...}
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"원본 데이터 경로 없음: {raw_path}")

    # 이미 처리되어 있으면 스킵
    if (processed_path / "train").exists() and any((processed_path / "train").iterdir()):
        print(f"[데이터] {processed_path}에 이미 처리된 데이터가 있음. 스킵.")
        classes = sorted([d.name for d in (processed_path / "train").iterdir() if d.is_dir()])
        return {"classes": classes, "skipped": True}

    # 클래스 폴더 찾기
    all_class_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
    if target_classes:
        class_dirs = [d for d in all_class_dirs if d.name in target_classes]
        missing = set(target_classes) - {d.name for d in class_dirs}
        if missing:
            print(f"[경고] 다음 클래스 폴더 없음: {missing}")
    else:
        class_dirs = all_class_dirs

    if not class_dirs:
        raise ValueError(f"클래스 폴더를 찾을 수 없음. raw_dir 구조 확인 필요: {raw_path}")

    print(f"[데이터] 사용할 클래스 {len(class_dirs)}개: {[d.name for d in class_dirs]}")

    # 분할 비율 검증
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율 합계가 1이어야 함"

    random.seed(seed)
    total_train, total_val, total_test = 0, 0, 0

    for class_dir in class_dirs:
        # 지원 확장자
        images = [
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if not images:
            print(f"[경고] {class_dir.name}에 이미지 없음, 스킵")
            continue

        random.shuffle(images)
        if max_images_per_class:
            images = images[:max_images_per_class]

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split_name, split_images in splits.items():
            dst_dir = processed_path / split_name / class_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, dst_dir / img_path.name)

        total_train += len(splits["train"])
        total_val += len(splits["val"])
        total_test += len(splits["test"])
        print(f"  {class_dir.name}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    classes = sorted([d.name for d in class_dirs])
    return {
        "classes": classes,
        "num_train": total_train,
        "num_val": total_val,
        "num_test": total_test,
    }


def get_augmentation() -> tf.keras.Sequential:
    """학습용 데이터 증강 파이프라인.

    실사용 환경(어두운 식당, 흔들린 사진)에 robust 하도록 적당히 강하게 설정.
    너무 강하면 학습 어려워지므로 정확도 안 나오면 강도 낮추기.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="augmentation")


def build_dataset(
    data_dir: str,
    image_size: int,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    """tf.data.Dataset 생성.

    Args:
        data_dir: 이미지 폴더 (클래스별 하위 폴더 구조)
        image_size: 리사이즈 사이즈
        batch_size: 배치 크기
        shuffle: 셔플 여부 (학습 데이터만 True)
        augment: 데이터 증강 적용 여부 (학습 데이터만 True)
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        label_mode="categorical",
    )

    # 픽셀 값을 [0, 1]로 정규화
    # 주의: MobileNetV2의 preprocess_input은 [-1, 1]을 기대함
    # model.py에서 preprocess_input을 적용하므로 여기선 [0, 255] 유지
    # 만약 다른 모델 쓰면 model.py의 preprocess와 일치하는지 확인

    if augment:
        augmentation = get_augmentation()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    # 단독 실행: 데이터 준비만 수행
    from src.config import load_config

    config = load_config()
    info = prepare_dataset(
        raw_dir=config["data"]["raw_dir"],
        processed_dir=config["data"]["processed_dir"],
        target_classes=config["data"]["target_classes"] or None,
        max_images_per_class=config["data"]["max_images_per_class"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config["misc"]["random_seed"],
    )
    print(f"\n[완료] 데이터 준비됨: {info}")
