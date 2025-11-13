#!/bin/bash
# IHS 데이터 빠른 테스트 스크립트
# 작은 샘플(1개 파일)로 전체 파이프라인 테스트

set -e

echo "========================================="
echo "IHS Quick Test Pipeline"
echo "========================================="

# 1. 작은 샘플 데이터 전처리 (ihs06.csv 하나만)
echo ""
echo "Step 1: Preprocessing sample data (ihs06.csv only)..."
python -c "
import pandas as pd
import shutil
from pathlib import Path

# Create temp directory
temp_dir = Path('data/ihsdata/temp_sample')
temp_dir.mkdir(parents=True, exist_ok=True)

# Copy one file
shutil.copy('data/ihsdata/raw/ihs06.csv', temp_dir / 'ihs06.csv')
print(f'✓ Copied ihs06.csv to {temp_dir}')
"

python preprocess_ihs.py \
    --input-dir data/ihsdata/temp_sample \
    --output data/ihsdata/processed/events_sample.csv

echo "✓ Preprocessing complete"

# 2. 센서 임베딩 학습 (빠른 테스트: 3 epochs)
echo ""
echo "Step 2: Training sensor embeddings (3 epochs)..."
python train/train_skipgram.py \
    --events-csv data/ihsdata/processed/events_sample.csv \
    --checkpoint checkpoint/ihs_sensor_embeddings_sample.pt \
    --embedding-dim 32 \
    --epochs 3 \
    --batch-size 2048

echo "✓ Embedding training complete"

# 3. 모델 학습 (빠른 테스트: 3 epochs, small settings)
echo ""
echo "Step 3: Training PV model (3 epochs, small settings)..."
python train/train_pv_ihs.py \
    --events-csv data/ihsdata/processed/events_sample.csv \
    --embeddings checkpoint/ihs_sensor_embeddings_sample.pt \
    --checkpoint checkpoint/pv_model_ihs_sample.pt \
    --window-size 50 \
    --stride 10 \
    --batch-size 16 \
    --epochs 3 \
    --device cuda \
    --hidden 64 \
    --mmu-hid 64 \
    --cmu-hid 64

echo ""
echo "========================================="
echo "✅ Quick test complete!"
echo "========================================="
echo ""
echo "Outputs:"
echo "  - Processed data: data/ihsdata/processed/events_sample.csv"
echo "  - Embeddings: checkpoint/ihs_sensor_embeddings_sample.pt"
echo "  - Model: checkpoint/pv_model_ihs_sample.pt"
echo ""
echo "To run on full dataset:"
echo "  1. Use data/ihsdata/processed/events.csv (already created)"
echo "  2. Increase --epochs to 20-50"
echo "  3. Increase --hidden, --mmu-hid, --cmu-hid to 128"
echo ""
