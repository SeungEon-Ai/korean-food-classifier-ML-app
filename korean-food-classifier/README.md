# 한국 음식 이미지 분류기 (Korean Food Classifier)

AI Hub의 한국 음식 데이터셋으로 학습한 이미지 분류 모델. 모바일 앱 배포(TFLite)를 목표로 함.

## 빠른 시작

### 1. 환경 설정
```bash
# Python 3.10+ 권장
pip install -r requirements.txt
```

### 2. 데이터 준비
AI Hub에서 다운받은 데이터를 `data/raw/` 폴더에 압축 해제.

폴더 구조는 다음 중 하나여야 함:
```
data/raw/
├── 김치찌개/
│   ├── img_001.jpg
│   └── ...
├── 된장찌개/
│   └── ...
```

다른 구조라면 `src/data_loader.py`의 `prepare_dataset()` 수정.

### 3. 설정 확인
`config.yaml` 파일에서 클래스 수, 학습 epoch 등 조정.

### 4. 학습 실행
```bash
python -m src.train
```

### 5. 평가
```bash
python -m src.evaluate
```

### 6. TFLite 변환 (모바일 배포용)
```bash
python -m src.convert_tflite
```

## 프로젝트 구조

| 파일 | 역할 | 수정 빈도 |
|------|------|-----------|
| `config.yaml` | 모든 설정 (경로, 하이퍼파라미터) | ⭐⭐⭐ 자주 |
| `src/data_loader.py` | 데이터 로딩/전처리 | ⭐⭐ 가끔 |
| `src/model.py` | 모델 아키텍처 | ⭐⭐ 가끔 |
| `src/train.py` | 학습 루프 | ⭐ 드물게 |
| `src/evaluate.py` | 평가 메트릭 | ⭐ 드물게 |
| `src/convert_tflite.py` | TFLite 변환 | ⭐ 드물게 |

## 주요 수정 포인트

### 클래스 수 변경
`config.yaml`의 `data.target_classes` 수정 (빈 리스트면 전체 사용)

### 모델 변경 (MobileNetV2 → EfficientNet 등)
`config.yaml`의 `model.architecture` 수정

### 학습률/배치 사이즈 조정
`config.yaml`의 `training` 섹션 수정

## TODO
- [ ] 데이터 증강 강도 실험
- [ ] EfficientNet-Lite 비교
- [ ] 양자화 모델 정확도 측정
- [ ] 영양정보 DB 매칭 로직 추가
