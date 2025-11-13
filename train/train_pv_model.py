#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position-Velocity-MMU/CMU Model Training Script
================================================

전체 파이프라인:
1. RichFeatureExtractor로 features 추출
2. PVDataset으로 변환
3. SmartHomeModel 학습
4. MultiTaskLoss로 다중 과제 최적화

사용법:
    python train/train_pv_model.py \
        --events-csv data/processed/events.csv \
        --embeddings checkpoint/sensor_embeddings_32d.pt \
        --checkpoint checkpoint/pv_model.pt \
        --window-size 100 \
        --stride 10 \
        --batch-size 32 \
        --epochs 50 \
        --learning-rate 3e-4 \
        --device cuda
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

# Import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.rich_features import RichFeatureExtractor
from model.pv_dataset import PVDataset, collate_pv_features, create_pv_datasets
from model.position_velocity_model import SmartHomeModel, MultiTaskLoss


def build_vocab(events: pd.DataFrame) -> Tuple[Dict, Dict]:
    """센서/활동 vocabulary 구축"""
    sensors = sorted(events['sensor'].unique())
    activities = sorted(events['activity'].dropna().unique())
    
    sensor_vocab = {s: i for i, s in enumerate(sensors)}
    activity_vocab = {a: i for i, a in enumerate(activities)}
    
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
    for activity, group in events.groupby('activity'):
        if pd.isna(activity):
            continue
        
        group = group.sort_values('timestamp').reset_index(drop=True)
        features = extractor.extract_sequence(
            events=group,
            window_size=window_size,
            stride=stride
        )
        all_features.extend(features)
    
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
    
    for batch in loader:
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
        'f1_macro': f1_score(all_labels, all_preds, average='macro') * 100,
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted') * 100
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
    
    for batch in loader:
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
        'f1_macro': f1_score(all_labels, all_preds, average='macro') * 100,
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted') * 100
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
    
    # Vocabularies
    sensor_vocab, activity_vocab = build_vocab(events)
    print(f"   Sensors: {len(sensor_vocab)}, Activities: {len(activity_vocab)}")
    
    # Load sensor embeddings
    print(f"\n📦 Loading sensor embeddings from {args.embeddings}")
    emb_state = torch.load(args.embeddings, map_location='cpu')
    sensor_embeddings = emb_state['embeddings'].numpy()
    print(f"   Embedding shape: {sensor_embeddings.shape}")
    
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
    
    # Model
    print(f"\n🏗️  Building SmartHomeModel")
    # F_base = N_sensor*2 (frame+ema) + V (vel) + E (emb)
    F_base = len(sensor_vocab) * 2 + 6 + sensor_embeddings.shape[1]
    
    model = SmartHomeModel(
        num_sensors=len(sensor_vocab),
        base_feat_dim=F_base,
        sensor_emb_dim=sensor_embeddings.shape[1],
        vel_dim=args.vel_dim,
        enc_hid=args.hidden,
        mmu_hid=args.mmu_hid,
        cmu_hid=args.cmu_hid,
        n_classes=len(activity_vocab)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Loss & Optimizer
    criterion = MultiTaskLoss(
        lambda_move=args.lambda_move,
        lambda_pos=args.lambda_pos,
        lambda_smooth=args.lambda_smooth
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay  # Increased for regularization
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
    parser = argparse.ArgumentParser(description="Train Position-Velocity-MMU/CMU Model")
    
    # Data
    parser.add_argument('--events-csv', type=str, required=True,
                        help='Path to processed events.csv')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Path to sensor embeddings (.pt)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to save best model checkpoint')
    
    # Feature extraction
    parser.add_argument('--window-size', type=int, default=100,
                        help='Sequence window size')
    parser.add_argument('--stride', type=int, default=10,
                        help='Sliding window stride')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train/val split ratio')
    
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
