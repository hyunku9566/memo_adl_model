
# skipgram 생성
python train/train_skipgram.py \
  --events-csv data/processed/events.csv \
  --checkpoint checkpoint/sensor_embeddings_32d.pt \
  --embedding-dim 32 \
  --epochs 10

# 모델학습
python train/train_no_emb.py \
    --events-csv data/processed/events.csv \
    --checkpoint checkpoint/pv_model_no_emb.pt \
    --model-type noemb \
    --stride 5 \
    --epochs 100 \
    --seed 42

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