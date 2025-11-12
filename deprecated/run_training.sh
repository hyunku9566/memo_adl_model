#!/bin/bash
# EMA-Attention Adaptive Decay 모델 학습 스크립트 (aiot-gpu 환경용)

set -e

echo "=================================="
echo "🚀 학습 준비 중..."
echo "=================================="
echo ""

# 1. PyTorch 설치
echo "📦 [1/3] PyTorch (CUDA 11.8) 설치 중..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. 필수 라이브러리
echo "📦 [2/3] 필수 라이브러리 설치 중..."
pip install -q numpy pandas scikit-learn tqdm

# 3. 확인
echo "✅ [3/3] 설치 확인 중..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=================================="
echo "📊 학습 시작"
echo "=================================="
echo ""
echo "설정:"
echo "  - Events CSV: data/processed/events.csv"
echo "  - Checkpoint: checkpoint/adaptive_decay_model.pt"
echo "  - Window Size: 100"
echo "  - Batch Size: 128"
echo "  - Epochs: 30"
echo "  - Learning Rate: 3e-4"
echo "  - Device: CUDA"
echo ""

# 4. 학습 시작
cd /home/lee/memo_model_adl
python train/train_adaptive_decay_model.py \
  --events-csv data/processed/events.csv \
  --checkpoint checkpoint/adaptive_decay_model.pt \
  --window-size 100 \
  --batch-size 128 \
  --epochs 30 \
  --learning-rate 3e-4 \
  --device cuda

echo ""
echo "=================================="
echo "✅ 학습 완료!"
echo "=================================="
echo ""
echo "결과 확인:"
echo "  cat checkpoint/adaptive_decay_model.metrics.json | python -m json.tool"
