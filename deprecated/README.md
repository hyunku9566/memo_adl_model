# Deprecated Files

이 폴더는 EMA Adaptive Decay Memory 모델 (v2.0) 이전의 구버전 파일들을 보관합니다.

## ⚠️ 주의사항

**이 폴더의 파일들은 현재 파이프라인에서 사용되지 않습니다.**

현재 활성 파일은 프로젝트 루트의 `README.md`를 참고하세요.

## 📁 구조

```
deprecated/
├── model/                        # 구버전 모델 파일
│   ├── activity_model.py        # 초기 활동 분류 모델
│   ├── adaptive_decay_attention.py  # v1.x Adaptive Decay (단순 임베딩)
│   ├── sequence_dataset.py      # 단순 시퀀스 Dataset
│   ├── sequence_model.py        # Transformer 시퀀스 모델
│   └── features.py              # 기본 특징 추출
│
├── train/                        # 구버전 학습 스크립트
│   ├── train_activity_model.py  # 기본 활동 모델 학습
│   ├── train_adaptive_decay_model.py  # v1.x Adaptive Decay 학습
│   └── train_sequence_model.py  # Transformer 학습
│
├── docs/                         # 구버전 문서
│   ├── ADAPTIVE_DECAY_DETAILED.md  # v1.x 상세 설명
│   ├── ADAPTIVE_DECAY_MODEL.md     # v1.x 개요
│   ├── PREPROCESSING_PIPELINE.md   # v1.x 전처리
│   ├── QUICK_START_GUIDE.md        # v1.x 빠른 시작
│   └── model.md                    # 초기 모델 설명
│
└── run_training.sh               # 구버전 학습 스크립트
```

## 🔄 v1.x → v2.0 주요 변경사항

### 문제점 (v1.x)
- ❌ 단순 임베딩만 사용 (sensor_id → embedding)
- ❌ Rich features 부재 (EMA, velocity 등)
- ❌ 데이터 부족 (115 samples)
- ❌ 낮은 성능 (30.43% Val Acc)
- ❌ 심각한 과적합 (Train 58% vs Val 30%)

### 개선사항 (v2.0)
- ✅ **Rich Feature Pipeline**
  - X_frame: 이진 센서 상태
  - X_ema: EMA 평활화
  - X_vel: 속도/이동 특징
  - X_emb: Skip-gram 임베딩
- ✅ **더 많은 데이터** (1,149 samples, stride=10)
- ✅ **높은 성능** (87.83% Val Acc)
- ✅ **건강한 학습** (Train-Val gap 8.36%)
- ✅ **조건부 λ 학습** (이동/정지 상태 고려)

## 📊 성능 비교

```
┌──────────────┬─────────┬──────────┬──────────┬─────────────┐
│ Version      │ Samples │ Train Acc│ Val Acc  │ Train-Val Gap│
├──────────────┼─────────┼──────────┼──────────┼─────────────┤
│ v1.x (old)   │   115   │  58.70%  │  30.43%  │   28.27%    │
│ v2.0 (현재)   │  1,149  │  96.19%  │  87.83%  │    8.36%    │
├──────────────┼─────────┼──────────┼──────────┼─────────────┤
│ Improvement  │  10x ↑  │  +37.5%  │  +57.4%  │   -19.9%    │
└──────────────┴─────────┴──────────┴──────────┴─────────────┘
```

## 🗂️ 보관 이유

1. **개발 히스토리 추적**: 모델 발전 과정 기록
2. **코드 재사용**: 특정 컴포넌트 참고 가능
3. **비교 분석**: 성능 개선 효과 검증
4. **문제 해결**: 유사 문제 발생 시 참고

## 🚫 사용 금지

이 폴더의 파일들을 **새로운 학습이나 추론에 사용하지 마세요**.

대신 다음 파일들을 사용하세요:
- 모델: `model/ema_adaptive_decay.py`
- 특징: `model/rich_features.py`
- 학습: `train/train_ema_adaptive_decay.py`

## 📅 Deprecated Date

2025-11-12

---

**현재 버전 문서**: `/README.md`, `/EMBEDDING_ARCHITECTURE.md`
