"""원본 zip을 풀어서 train/val/test로 분리하고 tf.data.Dataset을 만든다."""
import random
import shutil
import zipfile
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

import config


def extract_zips(zip_dir=config.ZIP_DIR, out_dir=config.RAW_DIR):
    """zip_dir 안의 모든 zip을 out_dir에 푼다. 깨진 파일은 건너뛴다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for zip_path in sorted(Path(zip_dir).glob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(out_dir)
            print(f"{zip_path.name}: ok")
        except zipfile.BadZipFile:
            print(f"{zip_path.name}: bad zip")
            failed.append(zip_path.name)
    return failed


def flatten(raw_dir=config.RAW_DIR, flat_dir=config.FLAT_DIR):
    """raw/카테고리/카테고리/음식/*.jpg → flat/음식/*.jpg

    AI Hub 데이터셋이 한 단계 더 들어간 구조라서 평탄화 필요.
    """
    flat_dir.mkdir(parents=True, exist_ok=True)
    for category in Path(raw_dir).iterdir():
        if not category.is_dir():
            continue
        inner = category / category.name
        if not inner.is_dir():
            inner = category
        for food in inner.iterdir():
            if not food.is_dir():
                continue
            dst = flat_dir / food.name
            if dst.exists():
                continue
            shutil.move(str(food), str(dst))


def split_dataset(flat_dir=config.FLAT_DIR, out_dir=config.DATA_DIR, seed=config.SEED):
    """flat/음식/*.jpg → out_dir/{train,val,test}/음식/*.jpg

    심볼릭 링크로 만들어서 디스크 절약.
    """
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    classes = sorted(d for d in Path(flat_dir).iterdir() if d.is_dir())
    for cls in classes:
        images = [p for p in cls.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * config.TRAIN_RATIO)
        n_val = int(n * config.VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for name, files in splits.items():
            target = out_dir / name / cls.name
            target.mkdir(parents=True, exist_ok=True)
            for src in files:
                (target / src.name).symlink_to(src.resolve())

    return [c.name for c in classes]


augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])


def load(split, augment=False):
    """split: 'train' | 'val' | 'test'"""
    ds = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR / split,
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        shuffle=(split == "train"),
        label_mode="categorical",
        seed=config.SEED,
    )
    if augment:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
