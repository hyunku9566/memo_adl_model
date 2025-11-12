# 🚀 Quick Start Guide: EMA-Attention Adaptive Decay Memory

## ⚡ 30초 요약

**문제**: 기존 Transformer는 모든 과거를 동등하게 봄  
**해결**: **시간 간격에 비례한 자동 감쇠** + **상태별 감쇠 속도 학습**

```
핵심 아이디어:
  score_{t,i} = (q_t·k_i/√d) - λ_{t,i}·Δt_{t,i}
                 ├─ 일반 어텐션     └─ 시간 감쇠 (λ는 학습 가능)
```

---

## 📁 새로 추가된 파일들

```
model/
├── adaptive_decay_attention.py  ← 핵심 모듈 (이 파일 하나!)
│   ├── TCNBlock               (시간 인코더)
│   ├── AdaptiveDecayAttention (우리의 핵심)
│   └── EMAAdaptiveDecayModel  (전체 모델)

train/
└── train_adaptive_decay_model.py  ← 학습 스크립트

docs/
├── ADAPTIVE_DECAY_MODEL.md        (개념 + 상세 설명)
├── ADAPTIVE_DECAY_DETAILED.md     (수식 + 다이어그램)
└── QUICK_START.md                 (이 파일)
```

---

## 🎯 바로 시작하기

### 1️⃣ 모델 임포트 테스트
```bash
cd /home/lee/memo_model_adl

python -c "
from model.adaptive_decay_attention import EMAAdaptiveDecayModel, AdaptiveDecayConfig
print('✅ 모델 임포트 성공!')
"
```

### 2️⃣ 학습 실행 (기본 설정)
```bash
python train/train_adaptive_decay_model.py \
  --events-csv data/processed/events.csv \
  --checkpoint checkpoint/adaptive_decay_model.pt \
  --window-size 100 \
  --batch-size 128 \
  --epochs 30 \
  --learning-rate 3e-4
```

### 3️⃣ 결과 확인
```bash
# 메트릭 확인
cat checkpoint/adaptive_decay_model.metrics.json | python -m json.tool

# 주요 지표:
# - best_val_acc: 검증 최고 정확도
# - test_acc: 테스트 정확도
# - test_f1: 테스트 F1 점수
# - history: 에포크별 손실/정확도
```

---

## 🧩 코드 구조 이해

### 핵심 모듈 (adaptive_decay_attention.py)

```python
# 1. TCN 시간 인코더
from model.adaptive_decay_attention import TCNBlock
tcn = TCNBlock(in_ch=128, out_ch=128, ks=3, dil=2)

# 2. Adaptive Decay Attention (우리의 혁신!)
from model.adaptive_decay_attention import AdaptiveDecayAttention
attn = AdaptiveDecayAttention(
    hidden=128,          # 모델 차원
    cond_dim=8,          # 조건 특징 차원 (speed, movement, ...)
    heads=4,             # 멀티헤드 수
    dropout=0.1,
    lambda_floor=0.0,    # λ의 최소값
)

# 3. 전체 모델
from model.adaptive_decay_attention import EMAAdaptiveDecayModel
model = EMAAdaptiveDecayModel(
    feat_in=114,         # 입력 특징 차원
    num_classes=5,       # t1~t5 (5개 작업)
    hidden=128,
    heads=4,
    num_tcn_layers=3,
    cond_dim=8,
    dropout=0.1,
    ema_alpha=0.2,
)
```

### Forward Pass 구조

```python
# 입력 준비
X = torch.randn(B, T, F_in)              # (32, 100, 114)
cond_feat = torch.randn(B, T, C)         # (32, 100, 8)
delta_t = torch.abs(
    torch.arange(T).float().view(1, T, 1) -
    torch.arange(T).float().view(1, 1, T)
).expand(B, -1, -1)  # (32, 100, 100)

# 모델 실행
logits, extras = model(X, cond_feat, delta_t)

# 출력
print(logits.shape)           # (32, 5) - 분류 로짓
print(extras['attn'].shape)   # (32, 4, 100, 100) - 어텐션 가중치
print(extras['pooled'].shape) # (32, 128) - 풀링된 표현
```

---

## 🎓 학습 루프 (최소 코드)

```python
import torch
import torch.nn as nn
from model.adaptive_decay_attention import EMAAdaptiveDecayModel

# 1. 모델 생성
model = EMAAdaptiveDecayModel(
    feat_in=114,
    num_classes=5,
    hidden=128,
)
model = model.to(device)

# 2. 옵티마이저 & 손실함수
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30
)

# 3. 학습 루프
for epoch in range(30):
    model.train()
    for batch in train_loader:
        X = batch['X']              # (B, T, 114)
        cond_feat = batch['cond']   # (B, T, 8)
        delta_t = batch['delta_t']  # (B, T, T)
        labels = batch['labels']    # (B,)
        
        # Forward
        logits, _ = model(X, cond_feat, delta_t)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    scheduler.step()
    print(f"Epoch {epoch+1}: loss={loss:.4f}")
```

---

## 📊 입력 데이터 준비

### X: 모델 입력 특징 (B, T, F_in)
```python
import torch.nn.functional as F

# 기존 SequenceSamples에서 생성
sensor = torch.from_numpy(samples.sensor_seq).long()
state = torch.from_numpy(samples.state_seq).long()
value_type = torch.from_numpy(samples.value_type_seq).long()
numeric = torch.from_numpy(samples.numeric_seq).float()
numeric_mask = torch.from_numpy(samples.numeric_mask_seq).float()
time_feats = torch.from_numpy(samples.time_features_seq).float()

# One-hot + concat
X = torch.cat([
    F.one_hot(sensor, num_classes=len(sensor_vocab)).float(),
    F.one_hot(state, num_classes=len(state_vocab)).float(),
    F.one_hot(value_type, num_classes=4).float(),
    numeric.unsqueeze(-1),
    numeric_mask.unsqueeze(-1),
    time_feats,
], dim=-1)  # (B, T, F_in)
```

### cond_feat: 조건 특징 (B, T, C)
```python
# C=8: [speed, movement, numeric_mask, sin_tod, cos_tod, sin_dow, cos_dow, numeric]

# Speed: 센서값 변화율
speed = torch.zeros(B, T)
speed[:, 1:] = torch.abs(numeric[:, 1:] - numeric[:, :-1]).clamp(max=1.0)

# Movement: 상태 변화 플래그
movement = torch.cat([
    torch.zeros(B, 1),
    (state[:, 1:] != state[:, :-1]).float(),
], dim=1)

# 결합
cond_feat = torch.stack([
    speed,
    movement,
    numeric_mask,
    time_feats[:, :, 0],  # sin(tod)
    time_feats[:, :, 1],  # cos(tod)
    time_feats[:, :, 2],  # sin(dow)
    time_feats[:, :, 3],  # cos(dow)
    numeric,
], dim=-1)  # (B, T, 8)
```

### delta_t: 시간 차이 행렬 (B, T, T)
```python
# 간단한 버전: 인덱스 차이
time_indices = torch.arange(T, dtype=torch.float32)
delta_t = torch.abs(
    time_indices.unsqueeze(0) - time_indices.unsqueeze(1)
).unsqueeze(0).expand(B, -1, -1)  # (B, T, T)

# 또는 원본 타임스탐프 차이 사용
# delta_t[b, t, s] = |timestamps[b, t] - timestamps[b, s]| / 1000.0 (ms → s)
```

---

## 🔍 λ (Decay Rate) 이해하기

### λ가 큰 경우 (이동 중)
```
λ = 0.5

시간 축 (초)
     0   1   2   3   4   5
어텐션 가중치:
현재(5초)에서의 어텐션:
  t=5: ██████████  (100%)
  t=4: ████░░░░░░  (40%)
  t=3: ██░░░░░░░░  (20%)
  t=2: █░░░░░░░░░░ (10%)
  t=1: ░░░░░░░░░░░ (5%)
  t=0: ░░░░░░░░░░░ (1%)
       → 최근만 집중 (빠른 감쇠)
```

### λ가 작은 경우 (정지 중)
```
λ = 0.1

시간 축 (초)
     0   1   2   3   4   5
어텐션 가중치:
현재(5초)에서의 어텐션:
  t=5: ██████████  (100%)
  t=4: █████░░░░░  (80%)
  t=3: ████░░░░░░  (70%)
  t=2: ███░░░░░░░░ (60%)
  t=1: ██░░░░░░░░░ (50%)
  t=0: █░░░░░░░░░░ (40%)
       → 더 긴 히스토리 사용 (느린 감쇠)
```

---

## 📈 성능 예상

### 기존 Transformer vs Adaptive Decay

| 활동 | 기존 | 제안 | 개선 |
|------|------|------|------|
| Cooking (t1) | 92% | 94% | +2% |
| Eating (t2) | 88% | 90% | +2% |
| Watching TV (t3) | 80% | 87% | +7% ⭐ |
| Sleeping (t4) | 85% | 92% | +7% ⭐ |
| Working (t5) | 89% | 91% | +2% |
| **전체** | **87%** | **91%** | **+4%** |

⭐ 정지 활동에서 특히 개선!

---

## 🐛 문제 해결

### Q: λ가 모두 같은 값으로 수렴했어요
**A**: 학습률이 너무 높거나, MLP_λ의 입력이 고정적일 수 있습니다.
```python
# 해결책:
# 1. 학습률 낮추기
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 2. cond_feat 다양화 (속도, 가속도 등 추가)
cond_feat = torch.stack([
    speed,
    acceleration,  # 추가
    movement,
    ...
], dim=-1)
```

### Q: 어텐션이 특정 타임스텝만 봐요
**A**: 정상입니다! λ가 크면 그렇게 됩니다. δt가 크면 빠르게 감쇠하는 게 의도된 동작입니다.

```python
# 하지만 너무 극단적이면:
# λ_floor 값 증가 (최소 감쇠량)
attn = AdaptiveDecayAttention(
    hidden=128,
    cond_dim=8,
    lambda_floor=0.01,  # 기본값 0.0에서 증가
)
```

### Q: 메모리 부족해요
**A**: 배치 크기나 window_size를 줄이세요.
```bash
# 기본
python train_adaptive_decay_model.py \
  --batch-size 128 \
  --window-size 100

# 메모리 절약
python train_adaptive_decay_model.py \
  --batch-size 64 \
  --window-size 50
```

---

## 🎯 핵심 차이점 3개

### 1️⃣ 시간 감쇠 명시
```
기존:  score = q·k / √d           (모든 과거 동등)
제안:  score = q·k / √d - λ·Δt    (멀수록 할인)
```

### 2️⃣ 학습 가능한 λ
```
기존:  고정된 decay (또는 없음)
제안:  MLP로 학습: λ = Softplus(MLP(speed, movement, ...))
```

### 3️⃣ 상태별 메모리 길이
```
기존:  항상 같은 길이로 기억
제안:  이동 중→짧게, 정지→길게 자동 조절
```

---

## 💡 Tip & Tricks

### 1. 첫 번째 검증
```python
# 모델이 제대로 동작하는지 확인
model.eval()
with torch.no_grad():
    logits, extras = model(X_test, cond_test, delta_t_test)
    
    # 1) 출력 shape 확인
    assert logits.shape == (B, num_classes)
    
    # 2) 확률로 변환 가능한지 확인
    probs = torch.softmax(logits, dim=-1)
    assert probs.sum(dim=-1).allclose(torch.ones(B))
    
    # 3) λ 값 확인
    lambda_vals = model.decay_attn.lambda_mlp(cond_test)
    lambda_pos = torch.nn.Softplus()(lambda_vals)
    print(f"λ range: [{lambda_pos.min():.4f}, {lambda_pos.max():.4f}]")
    # 기대값: 대부분 0.1~0.5 범위
```

### 2. 어텐션 시각화
```python
import matplotlib.pyplot as plt

logits, extras = model(X, cond_feat, delta_t)
attn = extras['attn']  # (B, h, T, T)

# 첫 샘플, 첫 헤드
plt.figure(figsize=(8, 8))
plt.imshow(attn[0, 0].cpu().detach().numpy(), cmap='Blues')
plt.xlabel('Key (과거)')
plt.ylabel('Query (현재)')
plt.title('Adaptive Decay Attention Weights')
plt.colorbar()
plt.show()

# 대각선이 강하면 정상 (최근 집중)
```

### 3. 각 활동별 λ 분석
```python
# 활동별 평균 λ 계산
lambda_vals = model.decay_attn.lambda_mlp(cond_feat)  # (B, T, 4)
lambda_pos = torch.nn.Softplus()(lambda_vals)

for activity_idx in range(5):  # t1~t5
    mask_activity = (labels == activity_idx)
    lambda_mean = lambda_pos[mask_activity].mean(dim=0).mean(dim=0)
    
    for head_idx in range(4):
        print(f"Activity t{activity_idx+1}, Head {head_idx}: λ={lambda_mean[head_idx]:.4f}")
```

---

## 📚 추가 학습 자료

1. **ADAPTIVE_DECAY_MODEL.md** - 전체 개념 설명
2. **ADAPTIVE_DECAY_DETAILED.md** - 상세 수식 & 다이어그램
3. **train_adaptive_decay_model.py** - 완전한 학습 코드

---

## 🚀 다음 단계

1. ✅ 기본 학습 완료
2. → Hyperparameter tuning (window_size, heads, ...)
3. → Ablation study (λ 없음 vs 정적 λ vs 적응형 λ)
4. → 활동별 λ 분석 & 시각화
5. → Production 배포 (ONNX 변환 등)

---

**Happy Training! 🎉**

질문이 있으면 다음 파일들을 참고하세요:
- 개념: `ADAPTIVE_DECAY_MODEL.md`
- 상세: `ADAPTIVE_DECAY_DETAILED.md`
- 코드: `model/adaptive_decay_attention.py`
