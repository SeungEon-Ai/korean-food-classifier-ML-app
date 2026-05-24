# Korean Food Classifier

AI Hub 한식 이미지 데이터셋으로 학습한 분류기. MobileNetV2 기반, 모바일 배포(TFLite)용.

137개 한식 음식 분류 가능.

## 사용법

```bash
pip install -r requirements.txt
```

### 1. 데이터 준비

AI Hub에서 받은 zip을 `data/zips/` 에 둔 다음:

```python
from data import extract_zips, flatten, split_dataset

extract_zips()      # zip → data/raw/
flatten()           # data/raw/ → data/flat/  (음식별 폴더로 평탄화)
split_dataset()     # data/flat/ → data/split/{train,val,test}/
```

### 2. 학습

```bash
python train.py
```

설정은 `config.py`에서 바꾼다. 결과는 `models/best.keras` 와 `models/history.json`.

### 3. 평가

```bash
python evaluate.py
```

### 4. TFLite 변환

```bash
python export.py                  # dynamic (기본, 2.56MB)
python export.py --quant float16  # float16 (4.59MB)
python export.py --quant none     # 비양자화 (9.12MB)
```

### 5. 단일 이미지 예측

```bash
python predict.py path/to/image.jpg
```

## 구조

- `config.py` 모든 경로/하이퍼파라미터
- `data.py` zip 해제, 평탄화, split, tf.data 로더
- `model.py` 모델 정의 (MobileNetV2 + Rescaling)
- `train.py` 학습 루프
- `evaluate.py` 평가 + confusion matrix
- `export.py` TFLite 변환
- `predict.py` 단일 이미지 추론
- `notebooks/colab.ipynb` Colab에서 한 번에 돌리는 노트북

## 결과 (137 클래스)

| metric | value |
|---|---|
| 클래스 수 | 137 |
| 학습 이미지 | 약 96,000장 |
| 검증 이미지 | 약 20,000장 |
| 테스트 이미지 | 약 21,000장 |
| Top-1 accuracy | 57.37% |
| Top-3 accuracy | 77.14% |
| Top-5 accuracy | 84.00% |
| 학습 시간 | 9.61시간 (T4 GPU) |
| TFLite (dynamic) | 2.56 MB |

### 학습 곡선

![training curves](docs/training_curves.png)

25 epoch 전체 학습 완료. Train과 Val accuracy 곡선이 가까이 따라가 과적합 없음.

### Confusion Matrix

![confusion matrix](docs/confusion_matrix.png)

137 클래스의 혼동행렬. 대각선이 진할수록 정답률 높음.

### 모델 아키텍처

- Base: MobileNetV2 (ImageNet pretrained, frozen)
- Input: 224×224×3, [-1, 1] 범위로 정규화 (Rescaling layer)
- Classification head: GlobalAveragePooling2D → Dropout(0.3) → Dense(137, softmax)
- 파라미터: 약 2.4M

### 학습 설정

- Optimizer: Adam (lr=1e-3, ReduceLROnPlateau 적용)
- Loss: Categorical Crossentropy
- Batch size: 32
- Augmentation: RandomFlip, RandomRotation(0.15), RandomZoom(0.15), RandomContrast(0.2), RandomBrightness(0.2)
- Early Stopping: patience=5

### 데이터셋

- 출처: AI Hub - 한국 음식 이미지 데이터셋
- 24개 카테고리 (구이, 국, 김치, 면, 밥, 찌개, 탕 등)
- 137개 음식 클래스
- 클래스당 약 1000장 (총 137,174장)

## 모델 활용 권장

Top-1 정확도가 57%로 단일 답변은 한계가 있지만 Top-3가 77%, Top-5가 84%이므로 후보 제시형 UX 권장:

```
사용자: 사진 촬영
   ↓
앱: "이 중 무엇인가요?"
  1. 떡볶이 (45%)
  2. 라볶이 (23%)
  3. 마라떡볶이 (15%)
  [직접 입력]
```

## 라이선스

AI Hub 데이터는 비상업적 연구용. 학습된 모델 배포 시 라이선스 확인 필요.
