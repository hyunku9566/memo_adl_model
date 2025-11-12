#!/bin/bash
# Quick Start: Position-Velocity-MMU/CMU Model Training
# ======================================================

set -e

echo "🚀 Starting Position-Velocity-MMU/CMU Model Training"
echo "======================================================"
echo ""

# Conda environment

conda activate aiot-gpu

# Check Python
echo "🐍 Python version:"
python --version
echo ""

# Run training
echo "🎓 Starting training..."
echo ""

python train/train_pv_model.py \
    --events-csv data/processed/events.csv \
    --embeddings checkpoint/sensor_embeddings_32d.pt \
    --checkpoint checkpoint/pv_model.pt \
    --window-size 100 \
    --stride 10 \
    --batch-size 32 \
    --epochs 50 \
    --learning-rate 3e-4 \
    --vel-dim 32 \
    --hidden 128 \
    --mmu-hid 128 \
    --cmu-hid 128 \
    --lambda-move 1.0 \
    --lambda-pos 0.1 \
    --lambda-smooth 0.01 \
    --device cuda \
    --num-workers 0

echo ""
echo "✅ Training complete!"
echo "   Model saved to: checkpoint/pv_model.pt"
echo "   History saved to: checkpoint/pv_model.history.json"
