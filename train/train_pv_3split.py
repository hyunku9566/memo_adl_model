#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position-Velocity Model Training with Train/Val/Test Split
===========================================================

표준 3분할 방식:
- Train: 60%
- Validation: 20% (하이퍼파라미터 튜닝, 모델 선택)
- Test: 20% (최종 성능 평가, 한 번만 사용)

사용법:
    python train/train_pv_3split.py \
        --events-csv data/processed/events.csv \
        --embeddings checkpoint/sensor_embeddings.pt \
        --checkpoint checkpoint/pv_model_3split.pt \
        --stride 5 \
        --epochs 100 \
        --seed 42
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


def extract_features_3split(
    events: pd.DataFrame,
    sensor_vocab: Dict,
    activity_vocab: Dict,
    sensor_embeddings: np.ndarray,
    window_size: int,
    stride: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List, List, List]:
    """
    RichFeatures 추출 및 train/val/test 3분할
    
    Returns:
        train_features, val_features, test_features
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}"
    
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
    
    # Train/val/test split with seed
    np.random.seed(seed)
    np.random.shuffle(all_features)
    
    n_total = len(all_features)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_features = all_features[:n_train]
    val_features = all_features[n_train:n_train + n_val]
    test_features = all_features[n_train + n_val:]
    
    return train_features, val_features, test_features


def evaluate(model, loader, criterion, device, desc="Eval"):
    """모델 평가"""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Loss components
    loss_components = {'L_cls': 0.0, 'L_move': 0.0, 'L_pos': 0.0, 'L_smooth': 0.0}
    
    with torch.no_grad():
        for batch in loader:
            X_base = batch['X_base'].to(device)
            sensor_ids = batch['sensor_ids'].to(device)
            timestamps = batch['timestamps'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            logits, aux = model(X_base, sensor_ids, timestamps, return_aux=True)
            loss, loss_dict = criterion(
                logits, labels, aux, 
                model.pos_head.positions
            )
            
            # Accumulate
            total_loss += loss.item() * len(labels)
            for k in loss_components.keys():
                loss_components[k] += loss_dict[k] * len(labels)
            
            # Predictions
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = (all_preds == all_labels).mean() * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    
    metrics = {
        'loss': total_loss / len(all_labels),
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        **{k: v / len(all_labels) for k, v in loss_components.items()}
    }
    
    return metrics


def main(args):
    print("=" * 80)
    print("🚀 Training Position-Velocity Model (Train/Val/Test Split)")
    print("=" * 80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    # Load data
    print(f"\n📂 Loading events: {args.events_csv}")
    events = pd.read_csv(args.events_csv)
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    print(f"   Total events: {len(events):,}")
    
    # Build vocab
    sensor_vocab, activity_vocab = build_vocab(events)
    print(f"\n📚 Vocabularies:")
    print(f"   Sensors: {len(sensor_vocab)}")
    print(f"   Activities: {len(activity_vocab)} - {list(activity_vocab.keys())}")
    
    # Load embeddings
    print(f"\n📦 Loading embeddings: {args.embeddings}")
    emb_data = torch.load(args.embeddings, map_location='cpu')
    sensor_embeddings = emb_data['embeddings'].numpy()
    print(f"   Shape: {sensor_embeddings.shape}")
    
    # Extract features with 3-way split
    print(f"\n🔧 Extracting rich features (window={args.window_size}, stride={args.stride}, seed={args.seed})")
    print(f"   Split ratio: Train {args.train_ratio:.0%} / Val {args.val_ratio:.0%} / Test {args.test_ratio:.0%}")
    
    train_features, val_features, test_features = extract_features_3split(
        events, sensor_vocab, activity_vocab, sensor_embeddings,
        args.window_size, args.stride,
        args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    print(f"   Train samples: {len(train_features):,}")
    print(f"   Val samples: {len(val_features):,}")
    print(f"   Test samples: {len(test_features):,}")
    print(f"   Total: {len(train_features) + len(val_features) + len(test_features):,}")
    
    # Create datasets
    train_ds = PVDataset(train_features, sensor_vocab, activity_vocab)
    val_ds = PVDataset(val_features, sensor_vocab, activity_vocab)
    test_ds = PVDataset(test_features, sensor_vocab, activity_vocab)
    
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
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_pv_features,
        num_workers=args.num_workers
    )
    
    # Build model
    print(f"\n🏗️  Building SmartHomeModel")
    sample_feat = train_features[0]
    base_feat_dim = (
        sample_feat.X_frame.shape[1] +
        sample_feat.X_ema.shape[1] +
        sample_feat.X_vel.shape[1] +
        sample_feat.X_emb.shape[1]
    )
    
    model = SmartHomeModel(
        num_sensors=len(sensor_vocab),
        base_feat_dim=base_feat_dim,
        sensor_emb_dim=sensor_embeddings.shape[1],
        vel_dim=args.vel_dim,
        enc_hid=args.hidden,
        mmu_hid=args.mmu_hid,
        cmu_hid=args.cmu_hid,
        n_classes=len(activity_vocab)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Loss & optimizer
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
    print(f"\n🎓 Training for {args.epochs} epochs")
    print("=" * 80)
    
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            X_base = batch['X_base'].to(device)
            sensor_ids = batch['sensor_ids'].to(device)
            timestamps = batch['timestamps'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits, aux = model(X_base, sensor_ids, timestamps, return_aux=True)
            loss, loss_dict = criterion(
                logits, labels, aux,
                model.pos_head.positions
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Train metrics
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        train_acc = (train_preds == train_labels).mean() * 100
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0) * 100
        train_loss = train_loss / len(train_labels)
        
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device, "Val")
        
        # Scheduler step
        scheduler.step(val_metrics['f1_macro'])
        
        # Time
        elapsed = time.time() - t_start
        
        # Print
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% F1: {train_f1:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.2f}% F1: {val_metrics['f1_macro']:.2f}% | "
              f"Time: {elapsed:.1f}s")
        
        # Save best model
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1_macro': val_metrics['f1_macro'],
                'val_acc': val_metrics['acc'],
                'sensor_vocab': sensor_vocab,
                'activity_vocab': activity_vocab,
                'args': vars(args)
            }, args.checkpoint)
            
            print(f"  ✓ Saved best model (Val F1: {val_metrics['f1_macro']:.2f}%, Acc: {val_metrics['acc']:.2f}%)")
        
        # Print detailed metrics every 10 epochs
        if epoch % 10 == 0:
            print(f"  └─ Train: L_cls={loss_dict['L_cls']:.4f} "
                  f"L_move={loss_dict['L_move']:.4f} "
                  f"L_pos={loss_dict['L_pos']:.4f} "
                  f"L_smooth={loss_dict['L_smooth']:.4f}")
            print(f"  └─ Val:   L_cls={val_metrics['L_cls']:.4f} "
                  f"L_move={val_metrics['L_move']:.4f} "
                  f"L_pos={val_metrics['L_pos']:.4f} "
                  f"L_smooth={val_metrics['L_smooth']:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping triggered! No improvement for {args.patience} epochs.")
            print(f"   Best epoch: {best_epoch} (Val F1: {best_val_f1:.2f}%)")
            break
        
        # History
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1_macro': train_f1,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_f1_weighted': val_metrics['f1_weighted']
        })
    
    # Save history
    history_path = Path(args.checkpoint).with_suffix('.history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final Test Evaluation (only once!)
    print("\n" + "=" * 80)
    print("🧪 FINAL TEST EVALUATION (사용 한 번만!)")
    print("=" * 80)
    
    # Load best model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, "Test")
    
    print(f"\n📊 Test Set Results:")
    print(f"   Accuracy: {test_metrics['acc']:.2f}%")
    print(f"   F1 Macro: {test_metrics['f1_macro']:.2f}%")
    print(f"   F1 Weighted: {test_metrics['f1_weighted']:.2f}%")
    print(f"   Loss: {test_metrics['loss']:.4f}")
    
    # Save test results
    test_results = {
        'test_acc': test_metrics['acc'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_weighted': test_metrics['f1_weighted'],
        'test_loss': test_metrics['loss'],
        'best_val_f1_macro': best_val_f1,
        'best_epoch': best_epoch,
        'n_train': len(train_features),
        'n_val': len(val_features),
        'n_test': len(test_features)
    }
    
    test_path = Path(args.checkpoint).with_suffix('.test_results.json')
    with open(test_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Training complete!")
    print(f"   Best Val F1 (Macro): {best_val_f1:.2f}% at epoch {best_epoch}")
    print(f"   Test F1 (Macro): {test_metrics['f1_macro']:.2f}%")
    print(f"   Model saved to: {args.checkpoint}")
    print(f"   History saved to: {history_path}")
    print(f"   Test results saved to: {test_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with Train/Val/Test 3-way split")
    
    # Data
    parser.add_argument('--events-csv', type=str, required=True)
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    
    # Feature extraction
    parser.add_argument('--window-size', type=int, default=100)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--train-ratio', type=float, default=0.6,
                        help='Training set ratio (default: 0.6 = 60%)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2 = 20%)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2 = 20%)')
    
    # Model architecture
    parser.add_argument('--vel-dim', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--mmu-hid', type=int, default=128)
    parser.add_argument('--cmu-hid', type=int, default=128)
    
    # Loss weights
    parser.add_argument('--lambda-move', type=float, default=0.5)
    parser.add_argument('--lambda-pos', type=float, default=0.1)
    parser.add_argument('--lambda-smooth', type=float, default=0.01)
    
    # Training
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)
