#!/bin/bash
# aiot-gpu 환경 설정 스크립트

echo "🚀 aiot-gpu 환경 설정 시작..."

# 환경 활성화
conda activate aiot-gpu

# PyTorch 설치 (GPU 지원)
echo "📦 PyTorch (CUDA 11.8) 설치 중..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 필요한 라이브러리 설치
echo "📦 필수 라이브러리 설치 중..."
pip install numpy pandas scikit-learn tensorboard tqdm

echo "✅ 설정 완료!"
echo ""
echo "🎓 학습 시작:"
echo "  python train/train_adaptive_decay_model.py --epochs 30"
