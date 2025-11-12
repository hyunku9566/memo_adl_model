# Smart Home Activity Recognition Models

**두 가지 혁신적인 아키텍처 제공:**
1. **EMA Adaptive Decay Attention** (87.83% Val Acc) - Attention에 time decay 적용 🆕
2. **Position-Velocity-MMU/CMU** (New!) - 학습 가능한 센서 위치 + 이중 메모리 🆕🆕

## 📁 프로젝트 구조

```
memo_model_adl/
├── 📊 data/
│   ├── raw/                      # 원본 CASAS 센서 데이터
│   │   └── p*.t*.csv            # 참가자별, 태스크별 파일
│   └── processed/
│       └── events.csv           # 통합된 이벤트 스트림 (11,586 events)
│
├── 🧠 model/                     # *** 활성 모델 모듈 ***
│   ├── data.py                  # 데이터 로딩 유틸리티
│   ├── skipgram.py              # Skip-gram 센서 임베딩
│   ├── rich_features.py         # Rich feature 추출기 ✨
│   ├── rich_dataset.py          # PyTorch Dataset wrapper ✨
│   ├── ema_adaptive_decay.py    # EMA Adaptive Decay 모델 ✨
│   ├── position_velocity_model.py  # Position-Velocity-MMU/CMU 모델 🆕
│   └── pv_dataset.py            # PV 모델용 Dataset 🆕
│
├── 🎓 train/                     # *** 활성 학습 스크립트 ***
│   ├── train_skipgram.py        # Skip-gram 임베딩 사전 학습
│   ├── train_ema_adaptive_decay.py  # EMA Adaptive Decay 학습 ✨
│   └── train_pv_model.py        # Position-Velocity 학습 🆕
│
├── 💾 checkpoint/                # 학습된 모델 체크포인트
│   ├── sensor_embeddings_32d.pt # Skip-gram 임베딩 (32D)
│   ├── ema_adaptive_decay_stride10.pt  # 최고 성능 모델 ⭐
│   └── *.metrics.json           # 학습 메트릭
│
├── 🗂️ deprecated/                # *** 이전 버전 (사용 안 함) ***
│   ├── model/                   # 구버전 모델 파일들
│   ├── train/                   # 구버전 학습 스크립트
│   └── docs/                    # 구버전 문서
│
├── 📖 utils/                     # 유틸리티
│   └── profiling.py             # 성능 프로파일링
│
├── 📄 Documentation
│   ├── EMBEDDING_ARCHITECTURE.md  # 전체 임베딩 구조 설명 ⭐
│   ├── preprocess.py            # 데이터 전처리 스크립트
│   └── setup_aiot_gpu.sh        # 환경 설정 스크립트
│
└── README.md                     # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Conda 환경 생성 및 활성화
conda activate aiot-gpu

# 또는 setup 스크립트 실행
bash setup_aiot_gpu.sh
```

### 2. 데이터 전처리

```bash
# 원본 데이터를 통합 events.csv로 변환
python preprocess.py
```

### 3. Skip-gram 임베딩 학습 (사전 학습)

```bash
python train/train_skipgram.py \
  --events-csv data/processed/events.csv \
  --checkpoint checkpoint/sensor_embeddings_32d.pt \
  --embedding-dim 32 \
  --context-size 5 \
  --epochs 10
```

### 4. EMA Adaptive Decay 모델 학습

```bash
/home/lee/anaconda3/envs/aiot-gpu/bin/python train/train_ema_adaptive_decay.py \
  --events-csv data/processed/events.csv \
  --embeddings checkpoint/sensor_embeddings_32d.pt \
  --checkpoint checkpoint/ema_adaptive_decay.pt \
  --window-size 100 \
  --stride 10 \
  --batch-size 64 \
  --epochs 30 \
  --learning-rate 1e-3 \
  --dropout 0.2 \
  --device cuda \
  --num-workers 0
```

## 📊 모델 성능

```
최고 성능 모델: ema_adaptive_decay_stride10.pt

데이터셋:
├─ Total samples: 1,149 (stride=10)
├─ Train: 919 samples
└─ Val: 230 samples

성능:
├─ Train Accuracy: 96.19%
├─ Val Accuracy: 87.83% ⭐
├─ Train-Val Gap: 8.36% (건강한 수준)
└─ Parameters: 393,481

활동 분류:
├─ t1, t2, t3, t4, t5 (5 classes)
└─ Baseline (Random): 20%
```

## 🏗️ 모델 아키텍처

### Rich Feature Pipeline

```python
Raw Events (11,586)
    ↓ [Sliding Window: size=100, stride=10]
1,149 Samples
    ↓ [Feature Extraction]
Rich Features:
    ├─ X_frame (100×30): 이진 센서 상태
    ├─ X_ema (100×30): EMA 평활화 (α=0.6)
    ├─ X_vel (100×6): 속도/이동 특징
    │   ├ speed, delta_pos, movement_flag
    │   ├ ema_speed, local_delta_t, activation_count
    └─ X_emb (100×32): Skip-gram 센서 임베딩
    ↓ [Concatenation]
X (100×98) = 통합 특징 벡터
```

### EMA Adaptive Decay Model

```python
Input (B, T, 98)
    ↓
Linear Projection (98 → 128) + LayerNorm
    ↓
TCN Backbone (3 layers, dilation=1,2,4)
    ↓
Adaptive Decay Attention
│ score = (q·k/√d) - λ·Δt
│ λ = Softplus(MLP(cond_feat))  # 학습 가능한 감쇠율
    ↓
Temporal Pooling (mask-aware)
    ↓
Classifier MLP (128 → 5)
    ↓
Output (B, 5) = Activity predictions
```

### 핵심 혁신: Adaptive Decay Attention

```python
# 기본 어텐션
score = q·k / √d

# 시간 감쇠 추가
score = (q·k / √d) - λ·Δt

# λ는 조건부 학습
λ = MLP(cond_feat)
  ├─ 이동 중 (movement_flag=1): λ↑ → 최근 이벤트에 집중
  └─ 정지 시 (movement_flag=0): λ↓ → 긴 히스토리 유지
```

## 📖 상세 문서

- **EMBEDDING_ARCHITECTURE.md**: 전체 임베딩 구조 및 각 특징의 역할 설명
- **deprecated/docs/**: 이전 버전 문서 (참고용)

## 🛠️ 개발 히스토리

### v2.0 (현재) - EMA Adaptive Decay Memory ⭐
- Rich feature pipeline (X_frame, X_ema, X_vel, X_emb)
- Adaptive λ 학습
- 87.83% Val Accuracy

### v1.x (deprecated) - 초기 구현
- 단순 임베딩 기반 모델
- 30.43% Val Accuracy
- `deprecated/` 폴더로 이동

## 📦 의존성

```yaml
Python: 3.11
PyTorch: 2.x (CUDA 11.8)
numpy, pandas, scikit-learn
tqdm, tensorboard
```

## 🎯 향후 개선 방향

1. **더 많은 샘플**: stride를 5 또는 3으로 줄여 샘플 증가
2. **앙상블**: 여러 window_size 모델 결합
3. **Cross-validation**: K-fold로 robust 성능 검증
4. **Attention 시각화**: λ와 어텐션 가중치 분석
5. **실시간 추론**: 온라인 예측 파이프라인 구현

## 📞 문의

프로젝트 관련 문의사항은 이슈를 생성해주세요.

---

**Last Updated**: 2025-11-12  
**Best Model**: `checkpoint/ema_adaptive_decay_stride10.pt` (87.83% Val Acc)
