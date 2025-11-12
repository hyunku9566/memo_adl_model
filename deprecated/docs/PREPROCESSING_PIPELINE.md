# 프리프로세싱 파이프라인 & 모델 입력 분석

## 📊 전체 파이프라인 흐름

```
Raw CSV Files (data/raw/)
    ↓
    ├─ p01.t1.csv (4 columns: date, time, sensor, message)
    ├─ p01.t2.csv
    └─ ... (모든 participant × task 파일)
    ↓
[preprocess.py] - 정규화 및 병합
    ↓
events.csv (단일 시간순 정렬 파일)
    ↓
[load_events] - EventData 생성 (data.py)
    ↓
[build_sequence_samples] - Sliding window 샘플 생성 (sequence_dataset.py)
    ↓
SequenceSamples (모델 학습용)
    ↓
[SensorSequenceModel] (Transformer encoder)
    ↓
Activity 예측
```

---

## 1️⃣ 단계 1: 원본 데이터 (Raw CSV)

### 입력 형식
```csv
date,time,sensor,message
2008-02-27,12:43:27.416392,M08,ON
2008-02-27,12:43:27.8481,M07,ON
2008-02-27,12:43:28.487061,M09,ON
```

**특징:**
- 4개 열만 존재: `date`, `time`, `sensor`, `message`
- Activity 정보는 **파일명**에서 추출됨
  - 예: `p01.t1.csv` → `t1` (Task 1)
  - 예: `p01.t2.csv` → `t2` (Task 2)
- 여러 participant × 5개 task = 총 ~225개 파일

---

## 2️⃣ 단계 2: preprocess.py (정규화)

### 목적
- 모든 원본 CSV를 **시간순으로 정렬**하여 단일 파일로 병합
- 모든 이벤트를 **일관된 스키마**로 정규화

### 변환 로직

| 항목 | 처리 |
|------|------|
| **timestamp** | `date` + `time` → ISO format (마이크로초 포함) |
| **sensor** | 원본 그대로 (M07, M08, I08 등) |
| **value_raw** | `message` 원본 값 |
| **value_type** | message 분석 → `state`, `numeric`, `string`, `missing` |
| **value_state** | STATE_TOKENS (ON, OFF, OPEN, CLOSE, PRESENT, ABSENT 등) 추출 |
| **activity** | 📍 **파일명에서 추출** (핵심!) |

### 핵심 함수: `_extract_activity_from_filename()`
```python
# 파일명 패턴: p<person>.<task>.csv
# 예: p01.t1.csv → "t1" 추출
match = re.search(r'\.([tp]\d+)', source)
return match.group(1)  # "t1", "t2", ... 반환
```

### 출력 형식 (events.csv)
```csv
timestamp,sensor,value_raw,value_type,value_numeric,value_state,activity,activity_phase,source_file
2008-02-26T10:50:08.326396,M07,ON,state,,ON,t1,,p40.t1.csv
2008-02-26T10:50:08.584328,M08,ON,state,,ON,t1,,p40.t1.csv
```

**결과:**
- 11,586개 이벤트 (220개 불량 타임스탐프 제외)
- 시간순 정렬됨
- 모든 샘플에 activity 레이블 포함

---

## 3️⃣ 단계 3: load_events() (EventData 생성)

### 목적
- CSV 로드 및 **어휘(vocabulary) 구축**
- 모든 categorical 값을 **정수 ID로 변환**

### 생성되는 어휘 (Vocabularies)

```
sensor_vocab:       ["M01", "M02", ..., "I08", "asterisk"]  (N개 센서)
                    ↓ ID 0, 1, ..., N-1

state_vocab:        ["<NONE>", "ON", "OFF", "OPEN", ...]    (상태)
                    ↓ ID 0, 1, 2, 3, ...

value_type_vocab:   ["missing", "state", "numeric", "string"]
                    ↓ ID 0, 1, 2, 3 (고정)

activity_vocab:     ["<NONE>", "t1", "t2", "t3", "t4", "t5"]
                    ↓ ID 0, 1, 2, 3, 4, 5
```

### EventData 구조
```python
@dataclass
class EventData:
    timestamps: List[datetime]           # 11,586개
    sensor_ids: List[int]               # 0~N (센서 인덱스)
    state_ids: List[int]                # 0~M (상태 인덱스)
    value_type_ids: List[int]           # 0~3 (값 타입 인덱스)
    numeric_values: List[float]         # 수치값 (센서 측정치)
    has_numeric: List[int]              # 0 또는 1 (수치값 존재 여부)
    activity_ids: List[int]             # 1~5 (t1~t5, 0은 unlabeled)
    
    # 어휘
    sensor_vocab: List[str]
    state_vocab: List[str]
    value_type_vocab: List[str]
    activity_vocab: List[str]
```

---

## 4️⃣ 단계 4: build_sequence_samples() (Sliding Window)

### 목적
- 각 **labeled 이벤트를 중심**으로 **sliding window** 생성
- 고정 크기 sequence로 변환 (기본: window_size=50)

### 프로세스

1. **labeled 이벤트 필터링**
   - `activity_ids[idx] > 0` 인 이벤트만 선택
   - 약 11,586개 이벤트 중 대부분이 레이블 있음

2. **각 labeled 이벤트에서 window 추출**
   ```
   Event timeline:
   ... E[i-50] E[i-49] ... E[i-1] E[i] ...
                       └─────────────┘
                      window_size=50 events
                   (가장 최근 50개 이벤트)
   
   E[i] = labeled 이벤트 (y = activity_ids[i] - 1)
   ```

3. **Multi-modal 시퀀스 표현**
   
   각 window에서 50개 이벤트 × 다음 정보 추출:

   | 필드 | 형태 | 설명 |
   |------|------|------|
   | `sensor_seq` | (50,) int | 센서 ID 시퀀스 |
   | `state_seq` | (50,) int | 상태 ID 시퀀스 |
   | `value_type_seq` | (50,) int | 값 타입 ID 시퀀스 |
   | `numeric_seq` | (50,) float | 정규화된 수치값 |
   | `numeric_mask_seq` | (50,) float | 수치값 존재 마스크 |
   | `time_features_seq` | (50, 4) float | 시간 특징 (sin/cos) |
   | `labels` | () int | Activity 레이블 (0~4) |

### 시간 특징 (Time Features)
```python
# 각 이벤트마다 계산:
# ToD (Time of Day) - 하루 중 시간 순환 인코딩
tod_angle = 2π × (hour×60 + minute) / (24×60)
tod_sin = sin(tod_angle)
tod_cos = cos(tod_angle)

# DoW (Day of Week) - 요일 순환 인코딩
dow_angle = 2π × weekday / 7
dow_sin = sin(dow_angle)
dow_cos = cos(dow_angle)

# time_features = [tod_sin, tod_cos, dow_sin, dow_cos]
```

### 출력: SequenceSamples
```python
@dataclass
class SequenceSamples:
    sensor_seq: (N_samples, 50) int64
    state_seq: (N_samples, 50) int64
    value_type_seq: (N_samples, 50) int64
    numeric_seq: (N_samples, 50) float32 (정규화됨)
    numeric_mask_seq: (N_samples, 50) float32
    time_features_seq: (N_samples, 50, 4) float32
    labels: (N_samples,) int64
    
    window_size: int = 50
    sensor_vocab: List[str]
    state_vocab: List[str]
    value_type_vocab: List[str]
    label_names: List[str] = ["t1", "t2", "t3", "t4", "t5"]
```

---

## 5️⃣ 단계 5: 학습/검증/테스트 분할

```python
# 시간순 분할 (시계열 데이터이므로 랜덤 분할 X)
train: 80% (처음 80%)
val:   10% (중간 10%)
test:  10% (마지막 10%)
```

---

## 6️⃣ 단계 6: SensorSequenceModel (Transformer)

### 모델 입력 (배치당)

```python
{
    "sensor": (batch_size, 50) int64         # 센서 ID
    "state": (batch_size, 50) int64          # 상태 ID
    "value_type": (batch_size, 50) int64     # 값 타입 ID
    "numeric": (batch_size, 50) float32      # 정규화된 수치
    "numeric_mask": (batch_size, 50) float32 # 마스크
    "time": (batch_size, 50, 4) float32      # 시간 특징
    "label": (batch_size,) int64             # 0~4 (t1~t5)
}
```

### 모델 아키텍처 흐름

```
Input (batch_size, 50)
    ↓
[Embeddings] - 각 항목 임베딩
    ├─ sensor: (batch, 50, 64)
    ├─ state: (batch, 50, 16)
    └─ value_type: (batch, 50, 8)
    ↓
[Numeric projection] - 수치값 + 마스크 → 16D
    ↓
[Time projection] - sin/cos 특징 → 16D
    ↓
[Concatenation]
    → (batch, 50, 64+16+8+16+16) = (batch, 50, 120)
    ↓
[Linear projection to model_dim]
    → (batch, 50, 128)
    ↓
[Positional embedding 추가]
    ↓
[Transformer Encoder]
    - 4개 attention heads
    - 2개 layers
    - GELU activation
    - dropout=0.2
    ↓
[마지막 타임스텝 추출]
    → (batch, 128)
    ↓
[Classification Head]
    - LayerNorm
    - Linear (128 → 256)
    - GELU
    - Dropout
    - Linear (256 → 5)  # t1, t2, t3, t4, t5
    ↓
logits: (batch, 5)
```

### 총 임베딩/프로젝션 차원 계산

| 컴포넌트 | 차원 |
|---------|------|
| 센서 임베딩 | 64 |
| 상태 임베딩 | 16 |
| 값타입 임베딩 | 8 |
| 수치값 프로젝션 | 16 |
| 시간 특징 프로젝션 | 16 |
| **합계** | **120** |
| → 모델차원으로 변환 | 128 |

---

## 7️⃣ 학습 설정

### 하이퍼파라미터

| 파라미터 | 기본값 |
|---------|--------|
| window_size | 50 events |
| batch_size | 512 |
| epochs | 20 |
| learning_rate | 3e-4 |
| sensor_embed_dim | 64 |
| state_embed_dim | 16 |
| value_type_embed_dim | 8 |
| numeric_feature_dim | 16 |
| time_feature_dim | 16 |
| model_dim | 128 |
| ff_dim | 256 |
| num_heads | 4 |
| num_layers | 2 |
| dropout | 0.2 |

### 손실함수 & 최적화
```python
# 클래스 가중치 계산 (불균형 데이터 대응)
class_weights = compute_class_weight('balanced', classes, y_train)

loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=weight_decay
)
```

---

## 📈 데이터 통계

```
원본 파일:           ~225개 (45 participants × 5 tasks)
총 이벤트:          11,586개
정규화 후:          11,586개 (모두 레이블 있음)

Activity 분포:
  - t1 (Task 1): X개
  - t2 (Task 2): X개
  - t3 (Task 3): X개
  - t4 (Task 4): X개
  - t5 (Task 5): X개

학습/검증/테스트 분할:
  - 학습: ~11,586 × 80% = ~9,269개
  - 검증: ~11,586 × 10% = ~1,159개
  - 테스트: ~11,586 × 10% = ~1,159개
```

---

## 🔑 핵심 포인트

### 1. Activity 레이블 소스
- **파일명**에서 추출 (예: `p01.t1.csv` → `t1`)
- 정규 표현식: `\.([tp]\d+)` 패턴 매칭

### 2. Multi-modal 입력
- **센서 ID** (어떤 센서인가)
- **상태 값** (ON/OFF/OPEN 등)
- **수치값** (온도, 습도 등)
- **시간 특징** (시간, 요일)

### 3. Sliding Window 기반
- 각 labeled 이벤트마다 **최근 50개 이벤트**를 context로 사용
- 시각적으로 시계열 컨텍스트 캡처

### 4. 임베딩 기반 표현
- Categorical 값들 → 학습 가능한 임베딩으로 변환
- Numeric/temporal 값들 → 뉴럴넷으로 프로젝션

### 5. Transformer 아키텍처
- Self-attention으로 window 내 이벤트 간 관계 학습
- 마지막 이벤트(레이블된 이벤트)의 표현으로 분류

---

## 💾 체크포인트 저장

```
checkpoint/activity_transformer.pt
├─ model state_dict
├─ vocabularies (sensor, state, value_type, activity)
├─ numeric stats (mean, std)
├─ hyperparameters
└─ CLI arguments
```

---

## 🚀 전체 명령어

```bash
# 1. 전처리 (원본 CSV → events.csv)
python preprocess.py

# 2. 모델 학습 (선택: skip-gram 임베딩 초기화)
python train/train_sequence_model.py \
  --events-csv data/processed/events.csv \
  --checkpoint checkpoint/activity_transformer.pt \
  --window-size 50 \
  --batch-size 512 \
  --epochs 20 \
  --learning-rate 3e-4 \
  --sensor-embedding-checkpoint checkpoint/sensor_embeddings.pt
```
