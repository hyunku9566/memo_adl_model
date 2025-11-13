# IHS 데이터셋 학습 가이드

IHS (In-Home Sensing) 데이터셋을 사용한 스마트홈 활동 인식 모델 학습 가이드입니다.

## 데이터 구조

IHS 데이터는 `data/ihsdata/raw/` 디렉토리에 있으며, 다음 형식을 따릅니다:

```csv
date,time,sensor,value[,activity[=phase]]
2015-10-08,00:10:03.360213,MainDoor,20.5
2015-10-08,01:24:41.157990,BedroomABed,ON,Sleep
2015-10-08,01:24:43.783981,BedroomABed,OFF,Sleep="begin"
```

- **date, time**: 이벤트 발생 시각
- **sensor**: 센서 이름 (예: BedroomABed, KitchenATemperature 등)
- **value**: 센서 값 (ON/OFF 또는 숫자)
- **activity**: 활동 라벨 (옵션, 예: Sleep, Bed_Toilet_Transition)
- **phase**: 활동 단계 (옵션, "begin" 또는 "end")

## 학습 파이프라인

### 1단계: 데이터 전처리

IHS 원본 CSV 파일들을 통합된 `events.csv` 형식으로 변환합니다.

```bash
python preprocess_ihs.py \
    --input-dir data/ihsdata/raw \
    --output data/ihsdata/processed/events.csv
```

**출력**: `data/ihsdata/processed/events.csv`
- 시간순으로 정렬된 통합 이벤트 로그
- 표준화된 컬럼: timestamp, sensor, raw_value, value_type, numeric_value, state, activity, activity_phase, source_file

### 2단계: 센서 임베딩 학습 (선택사항)

Skip-gram 방식으로 센서 임베딩을 학습합니다. 이 단계는 선택사항이며, 생략할 경우 모델 학습 시 임베딩이 랜덤 초기화됩니다.

```bash
python train/train_skipgram.py \
    --events-csv data/ihsdata/processed/events.csv \
    --checkpoint checkpoint/ihs_sensor_embeddings.pt \
    --embedding-dim 32 \
    --epochs 10 \
    --context-size 5 \
    --batch-size 4096 \
    --learning-rate 0.01
```

**출력**: `checkpoint/ihs_sensor_embeddings.pt`
- 센서별 32차원 임베딩 벡터
- Skip-gram 학습 통계 및 메트릭

### 3단계: Position-Velocity 모델 학습

전처리된 데이터와 센서 임베딩을 사용해 활동 인식 모델을 학습합니다.

```bash
python train/train_pv_ihs.py \
    --events-csv data/ihsdata/processed/events.csv \
    --embeddings checkpoint/ihs_sensor_embeddings.pt \
    --checkpoint checkpoint/pv_model_ihs.pt \
    --window-size 100 \
    --stride 10 \
    --batch-size 32 \
    --epochs 50 \
    --learning-rate 3e-4 \
    --device cuda
```

**주요 옵션**:
- `--window-size`: 시퀀스 윈도우 크기 (기본: 100)
- `--stride`: 슬라이딩 윈도우 보폭 (기본: 10)
- `--train-ratio`: Train/Val 분할 비율 (기본: 0.8)
- `--patience`: Early stopping patience (기본: 15)
- `--embedding-dim`: 임베딩 차원 (임베딩 파일이 없을 때, 기본: 32)

**출력**:
- `checkpoint/pv_model_ihs.pt`: 최고 성능 모델 체크포인트
- `checkpoint/pv_model_ihs.history.json`: 학습 히스토리 (loss, accuracy, F1 등)

## 모델 구조

Position-Velocity 모델은 다음 구조를 가지며, IHS 데이터의 센서 수와 활동 수에 맞춰 자동으로 차원이 조정됩니다:

```
입력 차원 (F_base) = N_sensors × 2 + 6 + embedding_dim
                    = (센서 frame + EMA) + (속도 특징) + (임베딩)

예: 센서 50개, 임베딩 32차원
    F_base = 50×2 + 6 + 32 = 138
```

### 핵심 컴포넌트

1. **PositionHead**: 센서별 학습 가능한 2D 위치
2. **VelocityHead**: 위치 차분으로 속도/방향/이동 패턴 추출
3. **MMU (Movement Memory Unit)**: 이동 패턴 기억
4. **CMU (Context Memory Unit)**: 맥락/영역 기억
5. **GateAndTrigger**: 이동/맥락 동적 융합
6. **TemporalEncoder**: TCN + BiGRU + Attention
7. **Classifier**: 최종 활동 분류

## 학습 결과 확인

학습이 완료되면 다음 정보가 저장됩니다:

```python
# 체크포인트 로드 예시
import torch

ckpt = torch.load('checkpoint/pv_model_ihs.pt')

print(f"Best Epoch: {ckpt['epoch']}")
print(f"Val F1 (Macro): {ckpt['val_f1_macro']:.2f}%")
print(f"Val Accuracy: {ckpt['val_acc']:.2f}%")
print(f"Sensors: {ckpt['num_sensors']}")
print(f"Activities: {ckpt['num_activities']}")
print(f"Activity Classes: {list(ckpt['activity_vocab'].keys())}")
```

## 빠른 시작 (전체 파이프라인)

```bash
# 1. 데이터 전처리
python preprocess_ihs.py

# 2. 센서 임베딩 학습 (선택)
python train/train_skipgram.py \
    --events-csv data/ihsdata/processed/events.csv \
    --checkpoint checkpoint/ihs_sensor_embeddings.pt \
    --epochs 10

# 3. 모델 학습
python train/train_pv_ihs.py \
    --events-csv data/ihsdata/processed/events.csv \
    --embeddings checkpoint/ihs_sensor_embeddings.pt \
    --checkpoint checkpoint/pv_model_ihs.pt \
    --epochs 50 \
    --device cuda
```

## 문제 해결

### 임베딩 vocab 크기 불일치

센서 임베딩의 vocab 크기가 데이터와 다를 경우, 자동으로 랜덤 초기화됩니다:

```
⚠️  Warning: Embedding vocab size (30) != data vocab size (50)
Reinitializing embeddings to match data...
```

### GPU 메모리 부족

배치 크기 또는 윈도우 크기를 줄여보세요:

```bash
python train/train_pv_ihs.py \
    --batch-size 16 \
    --window-size 50 \
    --device cuda
```

### 데이터 클래스 불균형

활동별 샘플 수가 불균형한 경우, stride를 조정하거나 data augmentation을 고려하세요.

## 기존 CASAS 데이터와의 차이점

| 항목 | CASAS | IHS |
|------|-------|-----|
| 타임스탬프 | 단일 컬럼 | date + time 분리 |
| 활동 라벨 | 마지막 컬럼 | 활동="phase" 형식 |
| 센서 수 | ~30개 | ~50개 (데이터셋에 따라 다름) |
| 활동 수 | 7-11개 | 데이터셋에 따라 다름 |

전처리 스크립트(`preprocess_ihs.py`)가 이러한 차이를 자동으로 처리합니다.

## 참고 문서

- 기존 CASAS 데이터 학습: `train/train_pv_model.py` 참조
- 모델 구조 상세: `model/position_velocity_model.py`
- 전처리 상세: `preprocess.py` (CASAS 버전)
