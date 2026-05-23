from pathlib import Path

# 경로
DATA_DIR = Path("data/split")          # train/val/test 하위 폴더
MODEL_DIR = Path("models")
ZIP_DIR = Path("data/zips")            # 원본 zip
RAW_DIR = Path("data/raw")             # 압축 푼 곳
FLAT_DIR = Path("data/flat")           # 평탄화 결과

# 데이터
IMAGE_SIZE = 224
BATCH_SIZE = 32
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# 나머지는 test

# 모델
DROPOUT = 0.3

# 학습
EPOCHS = 20
LR = 1e-3
PATIENCE = 5

SEED = 42
