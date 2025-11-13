#!/bin/bash
# IHS 전체 데이터 학습 스크립트
# 최적화된 설정으로 전체 파이프라인 실행

set -e

echo "========================================="
echo "IHS Full Dataset Training Pipeline"
echo "========================================="

# 1. 전처리 (이미 완료되어 있다면 스킵)
if [ ! -f "data/ihsdata/processed/events.csv" ]; then
    echo ""
    echo "Step 1: Preprocessing full dataset..."
    python preprocess_ihs.py \
        --input-dir data/ihsdata/raw \
        --output data/ihsdata/processed/events.csv
    echo "✓ Preprocessing complete"
else
    echo ""
    echo "Step 1: Preprocessing already done, skipping..."
fi

# 2. 센서 임베딩 학습 (선택사항 - 이미 있으면 스킵)
if [ ! -f "checkpoint/ihs_sensor_embeddings.pt" ]; then
    echo ""
    echo "Step 2: Training sensor embeddings (10 epochs)..."
    python train/train_skipgram.py \
        --events-csv data/ihsdata/processed/events.csv \
        --checkpoint checkpoint/ihs_sensor_embeddings.pt \
        --embedding-dim 32 \
        --context-size 5 \
        --negatives 5 \
        --batch-size 4096 \
        --epochs 10 \
        --learning-rate 0.01
    echo "✓ Embedding training complete"
else
    echo ""
    echo "Step 2: Embeddings already exist, skipping..."
fi

# 3. 모델 학습 (전체 데이터 - 최적화 설정)
echo ""
echo "Step 3: Training Position-Velocity model on full dataset..."
echo "  - Window size: 100"
echo "  - Stride: 10 (더 많은 샘플로 정확도 향상)"
echo "  - Batch size: 32"
echo "  - Epochs: 200 (장시간 학습으로 최고 성능 달성)"
echo "  - Hidden dims: 128"
echo ""

python train/train_pv_ihs.py \
    --events-csv data/ihsdata/processed/events.csv \
    --embeddings checkpoint/ihs_sensor_embeddings.pt \
    --checkpoint checkpoint/pv_model_ihs_full.pt \
    --window-size 50 \
    --batch-size 2048 \
    --use-cache \
    --num-workers 8 \
    --stride 5 \
    --use-amp \
    --n-jobs -1 \
    --epochs 200 \
    --learning-rate 3e-4 \
    --weight-decay 5e-4 \
    --patience 15 \
    --hidden 128 \
    --mmu-hid 128 \
    --cmu-hid 128 \
    --vel-dim 32 \
    --device cuda

echo ""
echo "========================================="
echo "✅ Full training complete!"
echo "========================================="
echo ""
echo "Outputs:"
echo "  - Processed data: data/ihsdata/processed/events.csv"
echo "  - Embeddings: checkpoint/ihs_sensor_embeddings.pt"
echo "  - Model: checkpoint/pv_model_ihs_full.pt"
echo "  - History: checkpoint/pv_model_ihs_full.history.json"
echo ""
echo "To check results:"
echo "  python -c \"import torch; ckpt = torch.load('checkpoint/pv_model_ihs_full.pt'); print(f'Val F1: {ckpt[\\\"val_f1_macro\\\"]:.2f}%')\""
echo ""
