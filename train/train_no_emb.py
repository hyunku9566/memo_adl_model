#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-Embedding Position-Velocity Model Training
==============================================

Skip-gram 임베딩 없이 순수 Position-Velocity + MMU/CMU만으로 학습

사용법:
    python train/train_no_emb.py \
        --events-csv data/processed/events.csv \
        --checkpoint checkpoint/pv_model_no_emb.pt \
        --model-type noemb \
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
from model.position_velocity_model import MultiTaskLoss
from model.position_velocity_model_no_emb import (
    SmartHomeModelNoEmb,
    SmartHomeModelNoEmbMinimal
)
from model.position_velocity_model_light import (
    SmartHomeModelLight,
    SmartHomeModelTiny,
    SmartHomeModelMicro
)


def build_vocab(events: pd.DataFrame) -> Tuple[Dict, Dict]:
    """센서/활동 vocabulary 구축"""
    sensors = sorted(events['sensor'].unique())
    activities = sorted(events['activity'].dropna().unique())
    
    sensor_vocab = {s: i for i, s in enumerate(sensors)}
    activity_vocab = {a: i for i, a in enumerate(activities)}
    
    return sensor_vocab, activity_vocab


def extract_features_no_emb(
    events: pd.DataFrame,
    sensor_vocab: Dict,
    activity_vocab: Dict,
    window_size: int,
    stride: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List, List, List]:
    """
    임베딩 없이 RichFeatures 추출 및 train/val/test 3분할
    
    Returns:
        train_features, val_features, test_features
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # 임베딩 없이 extractor 생성
    extractor = RichFeatureExtractor(
        sensor_vocab=sensor_vocab,
        activity_vocab=activity_vocab,
        sensor_embeddings=None,  # 임베딩 제거!
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


class NoEmbDataset(torch.utils.data.Dataset):
    """
    임베딩 제거 버전 Dataset
    X_base에서 임베딩 부분을 제거
    """
    
    def __init__(self, features_list, sensor_vocab, activity_vocab):
        self.features_list = features_list
        self.sensor_vocab = sensor_vocab
        self.activity_vocab = activity_vocab
        self.num_sensors = len(sensor_vocab)
    
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx):
        feat = self.features_list[idx]
        
        # X_base: [X_frame | X_ema | X_vel] (임베딩 제외!)
        X_base = np.concatenate([
            feat.X_frame,   # [T, N_sensor]
            feat.X_ema,     # [T, N_sensor]
            feat.X_vel,     # [T, 6]
        ], axis=-1)
        
        # sensor_ids
        sensor_ids = np.argmax(feat.X_ema, axis=-1)
        
        # timestamps
        if feat.delta_t is not None and feat.delta_t.shape[0] > 0:
            timestamps = feat.delta_t[0, :]
            timestamps = timestamps / (timestamps.max() + 1e-6) * 100
        else:
            timestamps = np.arange(feat.valid_length, dtype=np.float32)
        
        return {
            'X_base': torch.from_numpy(X_base).float(),
            'sensor_ids': torch.from_numpy(sensor_ids).long(),
            'timestamps': torch.from_numpy(timestamps).float(),
            'label': torch.tensor(feat.label, dtype=torch.long),
            'length': torch.tensor(feat.valid_length, dtype=torch.long)
        }


def collate_fn(batch):
    """Batch collation"""
    X_base = torch.nn.utils.rnn.pad_sequence(
        [b['X_base'] for b in batch], batch_first=True
    )
    sensor_ids = torch.nn.utils.rnn.pad_sequence(
        [b['sensor_ids'] for b in batch], batch_first=True
    )
    timestamps = torch.nn.utils.rnn.pad_sequence(
        [b['timestamps'] for b in batch], batch_first=True
    )
    labels = torch.stack([b['label'] for b in batch])
    
    return {
        'X_base': X_base,
        'sensor_ids': sensor_ids,
        'timestamps': timestamps,
        'labels': labels
    }


def evaluate(model, loader, criterion, device, desc="Eval"):
    """모델 평가"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            X_base = batch['X_base'].to(device)
            sensor_ids = batch['sensor_ids'].to(device)
            timestamps = batch['timestamps'].to(device)
            labels = batch['labels'].to(device)
            
            logits, aux = model(X_base, sensor_ids, timestamps, return_aux=True)
            loss, _ = criterion(logits, labels, aux, model.pos_head.positions)
            
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    
    return {
        'loss': avg_loss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """1 epoch 학습"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(loader):
        X_base = batch['X_base'].to(device)
        sensor_ids = batch['sensor_ids'].to(device)
        timestamps = batch['timestamps'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits, aux = model(X_base, sensor_ids, timestamps, return_aux=True)
        loss, losses = criterion(logits, labels, aux, model.pos_head.positions)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    
    avg_loss = total_loss / total
    accuracy = correct / total * 100
    
    return avg_loss, accuracy


def main(args):
    print("="*80)
    print("No-Embedding Position-Velocity Model Training")
    print("="*80)
    print(f"Model type: {args.model_type}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load data
    print("📦 Loading events...")
    events = pd.read_csv(args.events_csv)
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    print(f"   Total events: {len(events)}")
    
    # Build vocabulary
    sensor_vocab, activity_vocab = build_vocab(events)
    print(f"   Sensors: {len(sensor_vocab)}")
    print(f"   Activities: {len(activity_vocab)}")
    
    # Extract features (without embeddings!)
    print("\n🔧 Extracting features (no embeddings)...")
    train_features, val_features, test_features = extract_features_no_emb(
        events, sensor_vocab, activity_vocab,
        args.window_size, args.stride,
        args.train_ratio, args.val_ratio, args.test_ratio,
        args.seed
    )
    print(f"   Train: {len(train_features)}")
    print(f"   Val: {len(val_features)}")
    print(f"   Test: {len(test_features)}")
    
    # Datasets
    train_ds = NoEmbDataset(train_features, sensor_vocab, activity_vocab)
    val_ds = NoEmbDataset(val_features, sensor_vocab, activity_vocab)
    test_ds = NoEmbDataset(test_features, sensor_vocab, activity_vocab)
    
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Build model
    base_feat_dim = len(sensor_vocab) * 2 + 6  # frame + ema + vel (NO EMBEDDING!)
    print(f"\n🏗️  Building model...")
    print(f"   Base feature dim: {base_feat_dim} (N*2 + 6, 임베딩 제외)")
    
    if args.model_type == 'noemb':
        model = SmartHomeModelNoEmb(
            num_sensors=len(sensor_vocab),
            base_feat_dim=base_feat_dim,
            vel_dim=32,
            enc_hid=128,
            mmu_hid=128,
            cmu_hid=128,
            n_classes=len(activity_vocab)
        )
    elif args.model_type == 'minimal':
        model = SmartHomeModelNoEmbMinimal(
            num_sensors=len(sensor_vocab),
            base_feat_dim=base_feat_dim,
            vel_dim=32,
            enc_hid=128,
            mmu_hid=128,
            cmu_hid=128,
            n_classes=len(activity_vocab)
        )
    elif args.model_type == 'light':
        model = SmartHomeModelLight(
            num_sensors=len(sensor_vocab),
            base_feat_dim=base_feat_dim,
            vel_dim=32,
            enc_hid=64,
            mmu_hid=64,
            cmu_hid=64,
            n_classes=len(activity_vocab)
        )
    elif args.model_type == 'tiny':
        model = SmartHomeModelTiny(
            num_sensors=len(sensor_vocab),
            base_feat_dim=base_feat_dim,
            vel_dim=16,
            enc_hid=32,
            mmu_hid=32,
            cmu_hid=32,
            n_classes=len(activity_vocab)
        )
    elif args.model_type == 'micro':
        model = SmartHomeModelMicro(
            num_sensors=len(sensor_vocab),
            base_feat_dim=base_feat_dim,
            vel_dim=8,
            enc_hid=16,
            mmu_hid=16,
            cmu_hid=16,
            n_classes=len(activity_vocab)
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Loss & optimizer
    criterion = MultiTaskLoss(
        lambda_move=0.5,
        lambda_pos=0.1,
        lambda_smooth=0.01
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=5e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Training
    print(f"\n🚀 Training started...")
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device, "Val")
        
        scheduler.step(val_metrics['f1'])
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'sensor_vocab': sensor_vocab,
                'activity_vocab': activity_vocab,
            }, args.checkpoint)
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: Train Acc={train_acc:.2f}% "
                  f"Val F1={val_metrics['f1']:.2f}% (best={best_val_f1:.2f}%)")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_f1': val_metrics['f1'],
            'val_acc': val_metrics['accuracy']
        })
        
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate on test
    print(f"\n📊 Loading best model (epoch {best_epoch})...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, "Test")
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Best Val F1:  {best_val_f1:.2f}% (epoch {best_epoch})")
    print(f"Test F1:      {test_metrics['f1']:.2f}%")
    print(f"Test Acc:     {test_metrics['accuracy']:.2f}%")
    print(f"Test Precision: {test_metrics['precision']:.2f}%")
    print(f"Test Recall:    {test_metrics['recall']:.2f}%")
    print(f"{'='*80}")
    
    # Save results
    results = {
        'model_type': args.model_type,
        'total_params': total_params,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'history': history
    }
    
    result_path = args.checkpoint.replace('.pt', '.results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--events-csv', type=str, default='data/processed/events.csv')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/pv_model_no_emb.pt')
    parser.add_argument('--model-type', type=str, default='noemb', 
                        choices=['noemb', 'minimal', 'light', 'tiny', 'micro'],
                        help='noemb: BiGRU+Attn(128), minimal: UniGRU+Attn(128), light: BiGRU+Attn(64), tiny: BiGRU+Attn(32), micro: BiGRU+Attn(16)')
    parser.add_argument('--window-size', type=int, default=100)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
