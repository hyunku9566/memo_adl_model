#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IHS 데이터셋용 Position-Velocity Model 학습 스크립트
===================================================

IHS 데이터셋의 센서/활동 수에 맞춰 자동으로 모델 차원을 조정합니다.

주요 차이점:
1. 센서 수가 CASAS보다 많음 (자동 감지)
2. 활동 클래스 수가 다름 (자동 감지)
3. 입력 차원 F_base = N_sensor*2 + 6 + emb_dim (자동 계산)

사용법:
    # 1) IHS 데이터 전처리
    python preprocess_ihs.py \
        --input-dir data/ihsdata/raw \
        --output data/ihsdata/processed/events.csv
    
    # 2) 센서 임베딩 학습 (옵션)
    python train/train_skipgram.py \
        --events-csv data/ihsdata/processed/events.csv \
        --checkpoint checkpoint/ihs_sensor_embeddings.pt \
        --embedding-dim 32 \
        --epochs 10
    
    # 3) 모델 학습
    python train/train_pv_ihs.py \
        --events-csv data/ihsdata/processed/events.csv \
        --embeddings checkpoint/ihs_sensor_embeddings.pt \
        --checkpoint checkpoint/pv_model_ihs.pt \
        --epochs 50
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# Import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.rich_features import RichFeatureExtractor
from model.pv_dataset import PVDataset, collate_pv_features, create_pv_datasets
from model.position_velocity_model import SmartHomeModel, MultiTaskLoss


def build_vocab(events: pd.DataFrame) -> Tuple[Dict, Dict]:
    """센서/활동 vocabulary 구축 (자동 감지)"""
    sensors = sorted(events['sensor'].unique())
    
    # 활동 필터링: 빈 문자열 제외
    activities = sorted([a for a in events['activity'].dropna().unique() if a.strip() != ''])
    
    sensor_vocab = {s: i for i, s in enumerate(sensors)}
    activity_vocab = {a: i for i, a in enumerate(activities)}
    
    print(f"\n📊 Vocabulary Statistics:")
    print(f"   Sensors: {len(sensor_vocab)} ({', '.join(list(sensors)[:5])}...)")
    print(f"   Activities: {len(activity_vocab)} ({', '.join(activities)})")
    
    return sensor_vocab, activity_vocab


def extract_features(
    events: pd.DataFrame,
    sensor_vocab: Dict,
    activity_vocab: Dict,
    sensor_embeddings: np.ndarray,
    window_size: int,
    stride: int,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List, List]:
    """
    RichFeatures 추출 및 train/val 분할
    
    Returns:
        train_features, val_features
    """
    extractor = RichFeatureExtractor(
        sensor_vocab=sensor_vocab,
        activity_vocab=activity_vocab,
        sensor_embeddings=sensor_embeddings,
        ema_alpha=0.6,
        time_scale=1.0
    )
    
    # 활동별로 그룹화
    all_features = []
    activity_counts = {}
    
    grouped = list(events.groupby('activity'))
    print(f"   Processing {len(grouped)} activities...")
    
    for activity, group in tqdm(grouped, desc="  Extracting features by activity"):
        # 빈 문자열 활동은 건너뛰기
        if pd.isna(activity) or activity.strip() == '':
            continue
        
        group = group.sort_values('timestamp').reset_index(drop=True)
        features = extractor.extract_sequence(
            events=group,
            window_size=window_size,
            stride=stride
        )
        all_features.extend(features)
        activity_counts[activity] = len(features)
    
    print(f"\n📦 Feature Extraction Statistics:")
    print(f"   Total samples: {len(all_features):,}")
    print(f"   Samples per activity:")
    for act, count in sorted(activity_counts.items(), key=lambda x: -x[1]):
        print(f"      {act}: {count:,}")
    
    # Train/val split with seed
    np.random.seed(seed)
    np.random.shuffle(all_features)
    split_idx = int(len(all_features) * train_ratio)
    train_features = all_features[:split_idx]
    val_features = all_features[split_idx:]
    
    return train_features, val_features


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Dict[str, float]:
    """한 에폭 학습"""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    loss_accum = {'L_cls': 0.0, 'L_move': 0.0, 'L_pos': 0.0, 'L_smooth': 0.0}
    
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="  Training", leave=False):
        X_base = batch['X_base'].to(device)
        sensor_ids = batch['sensor_ids'].to(device)
        timestamps = batch['timestamps'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        logits, aux = model(X_base, sensor_ids, timestamps, return_aux=True)
        
        # Loss
        loss, losses = criterion(logits, labels, aux, model.pos_head.positions)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * len(labels)
        pred = logits.argmax(dim=-1)
        total_correct += (pred == labels).sum().item()
        total_samples += len(labels)
        
        # Store for F1 calculation
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Accumulate sub-losses
        for k, v in losses.items():
            loss_accum[k] += v * len(labels)
    
    # Average
    metrics = {
        'loss': total_loss / total_samples,
        'acc': total_correct / total_samples * 100,
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100,
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    }
    
    for k in loss_accum:
        metrics[k] = loss_accum[k] / total_samples
    
    return metrics


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: str
) -> Dict[str, float]:
    """검증"""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    loss_accum = {'L_cls': 0.0, 'L_move': 0.0, 'L_pos': 0.0, 'L_smooth': 0.0}
    
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="  Validating", leave=False):
        X_base = batch['X_base'].to(device)
        sensor_ids = batch['sensor_ids'].to(device)
        timestamps = batch['timestamps'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        logits, aux = model(X_base, sensor_ids, timestamps, return_aux=True)
        
        # Loss
        loss, losses = criterion(logits, labels, aux, model.pos_head.positions)
        
        # Metrics
        total_loss += loss.item() * len(labels)
        pred = logits.argmax(dim=-1)
        total_correct += (pred == labels).sum().item()
        total_samples += len(labels)
        
        # Store for F1 calculation
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        for k, v in losses.items():
            loss_accum[k] += v * len(labels)
    
    metrics = {
        'loss': total_loss / total_samples,
        'acc': total_correct / total_samples * 100,
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100,
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    }
    
    for k in loss_accum:
        metrics[k] = loss_accum[k] / total_samples
    
    return metrics


def main(args):
    """메인 학습 루프"""
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    
    # Load data
    print(f"\n📂 Loading events from {args.events_csv}")
    events = pd.read_csv(args.events_csv)
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    print(f"   Total events: {len(events):,}")
    
    # Vocabularies (자동 감지)
    sensor_vocab, activity_vocab = build_vocab(events)
    num_sensors = len(sensor_vocab)
    num_activities = len(activity_vocab)
    
    # Load or initialize sensor embeddings
    if args.embeddings and Path(args.embeddings).exists():
        print(f"\n📦 Loading sensor embeddings from {args.embeddings}")
        emb_state = torch.load(args.embeddings, map_location='cpu')
        sensor_embeddings = emb_state['embeddings'].numpy()
        print(f"   Embedding shape: {sensor_embeddings.shape}")
        
        # 차원 검증
        if sensor_embeddings.shape[0] != num_sensors:
            print(f"   ⚠️  Warning: Embedding vocab size ({sensor_embeddings.shape[0]}) != data vocab size ({num_sensors})")
            print(f"   Reinitializing embeddings to match data...")
            sensor_embeddings = np.random.randn(num_sensors, args.embedding_dim).astype(np.float32) * 0.1
    else:
        print(f"\n⚠️  No embeddings found at {args.embeddings}, initializing randomly")
        sensor_embeddings = np.random.randn(num_sensors, args.embedding_dim).astype(np.float32) * 0.1
    
    emb_dim = sensor_embeddings.shape[1]
    
    # Extract features
    print(f"\n🔧 Extracting rich features (window={args.window_size}, stride={args.stride}, seed={args.seed})")
    train_features, val_features = extract_features(
        events, sensor_vocab, activity_vocab, sensor_embeddings,
        args.window_size, args.stride, args.train_ratio, args.seed
    )
    print(f"   Train samples: {len(train_features):,}")
    print(f"   Val samples: {len(val_features):,}")
    
    # Datasets
    train_ds, val_ds = create_pv_datasets(
        train_features, val_features, sensor_vocab, activity_vocab
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_pv_features,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_pv_features,
        num_workers=args.num_workers
    )
    
    # Model (자동 차원 계산)
    print(f"\n🏗️  Building SmartHomeModel")
    # F_base = N_sensor*2 (frame+ema) + 6 (velocity features) + E (embedding dim)
    F_base = num_sensors * 2 + 6 + emb_dim
    print(f"   Input feature dimension (F_base): {F_base}")
    print(f"      = {num_sensors}*2 (sensor frame+ema) + 6 (velocity) + {emb_dim} (embedding)")
    
    model = SmartHomeModel(
        num_sensors=num_sensors,
        base_feat_dim=F_base,
        sensor_emb_dim=emb_dim,
        vel_dim=args.vel_dim,
        enc_hid=args.hidden,
        mmu_hid=args.mmu_hid,
        cmu_hid=args.cmu_hid,
        n_classes=num_activities
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {n_params:,}")
    print(f"   Sensors: {num_sensors}, Activities: {num_activities}")
    
    # Loss & Optimizer
    criterion = MultiTaskLoss(
        lambda_move=args.lambda_move,
        lambda_pos=args.lambda_pos,
        lambda_smooth=args.lambda_smooth
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Training loop
    print(f"\n🎓 Training for {args.epochs} epochs (Early stopping patience: {args.patience})")
    print("=" * 80)
    
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        
        # Scheduler (based on val F1)
        scheduler.step(val_metrics['f1_macro'])
        
        elapsed = time.time() - t0
        
        # Log
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.2f}% F1: {train_metrics['f1_macro']:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.2f}% F1: {val_metrics['f1_macro']:.2f}% | "
              f"Time: {elapsed:.1f}s")
        
        # Detailed loss breakdown (every 10 epochs)
        if epoch % 10 == 0:
            print(f"  └─ Train: L_cls={train_metrics['L_cls']:.4f} "
                  f"L_move={train_metrics['L_move']:.4f} "
                  f"L_pos={train_metrics['L_pos']:.4f} "
                  f"L_smooth={train_metrics['L_smooth']:.4f}")
            print(f"  └─ Val:   L_cls={val_metrics['L_cls']:.4f} "
                  f"L_move={val_metrics['L_move']:.4f} "
                  f"L_pos={val_metrics['L_pos']:.4f} "
                  f"L_smooth={val_metrics['L_smooth']:.4f}")
        
        # Save best (based on F1 macro)
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_metrics['acc'],
                'val_f1_macro': val_metrics['f1_macro'],
                'val_f1_weighted': val_metrics['f1_weighted'],
                'val_loss': val_metrics['loss'],
                'sensor_vocab': sensor_vocab,
                'activity_vocab': activity_vocab,
                'num_sensors': num_sensors,
                'num_activities': num_activities,
                'F_base': F_base,
                'args': vars(args)
            }
            
            torch.save(checkpoint, args.checkpoint)
            print(f"  ✓ Saved best model (Val F1: {val_metrics['f1_macro']:.2f}%, Acc: {val_metrics['acc']:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⚠️  Early stopping triggered! No improvement for {args.patience} epochs.")
                print(f"   Best epoch: {best_epoch} (Val F1: {best_val_f1:.2f}%)")
                break
        
        # History
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['acc'],
            'train_f1_macro': train_metrics['f1_macro'],
            'train_f1_weighted': train_metrics['f1_weighted'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_f1_weighted': val_metrics['f1_weighted'],
            **{f'train_{k}': v for k, v in train_metrics.items() if k.startswith('L_')},
            **{f'val_{k}': v for k, v in val_metrics.items() if k.startswith('L_')}
        })
    
    # Save history
    history_path = Path(args.checkpoint).with_suffix('.history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Training complete!")
    print(f"   Best Val F1 (Macro): {best_val_f1:.2f}% at epoch {best_epoch}")
    print(f"   Model saved to: {args.checkpoint}")
    print(f"   History saved to: {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Position-Velocity Model on IHS Dataset")
    
    # Data
    parser.add_argument('--events-csv', type=str, 
                        default='data/ihsdata/processed/events.csv',
                        help='Path to processed IHS events.csv')
    parser.add_argument('--embeddings', type=str, 
                        default='checkpoint/ihs_sensor_embeddings.pt',
                        help='Path to sensor embeddings (.pt). If not found, will initialize randomly.')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoint/pv_model_ihs.pt',
                        help='Path to save best model checkpoint')
    
    # Feature extraction
    parser.add_argument('--window-size', type=int, default=100,
                        help='Sequence window size')
    parser.add_argument('--stride', type=int, default=10,
                        help='Sliding window stride')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train/val split ratio')
    parser.add_argument('--embedding-dim', type=int, default=32,
                        help='Sensor embedding dimension (used if initializing randomly)')
    
    # Model architecture
    parser.add_argument('--vel-dim', type=int, default=32,
                        help='Velocity embedding dimension')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Encoder hidden dimension')
    parser.add_argument('--mmu-hid', type=int, default=128,
                        help='MMU hidden dimension')
    parser.add_argument('--cmu-hid', type=int, default=128,
                        help='CMU hidden dimension')
    
    # Loss weights
    parser.add_argument('--lambda-move', type=float, default=1.0,
                        help='Movement auxiliary loss weight')
    parser.add_argument('--lambda-pos', type=float, default=0.1,
                        help='Position regularization weight')
    parser.add_argument('--lambda-smooth', type=float, default=0.01,
                        help='Velocity smoothness weight')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay for AdamW (L2 regularization)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)
