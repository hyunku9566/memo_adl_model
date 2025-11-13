#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Set Evaluation Script
===========================

학습이 완료된 모델로 Test Set만 평가합니다.

사용법:
    python evaluate_test.py \
        --checkpoint checkpoint/pv_model_3split_seed42.pt \
        --events-csv data/processed/events.csv \
        --embeddings checkpoint/sensor_embeddings.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent))

from model.rich_features import RichFeatureExtractor
from model.pv_dataset import PVDataset, collate_pv_features
from model.position_velocity_model import SmartHomeModel, MultiTaskLoss


def build_vocab(events: pd.DataFrame):
    """센서/활동 vocabulary 구축"""
    sensors = sorted(events['sensor'].unique())
    activities = sorted(events['activity'].dropna().unique())
    
    sensor_vocab = {s: i for i, s in enumerate(sensors)}
    activity_vocab = {a: i for i, a in enumerate(activities)}
    
    return sensor_vocab, activity_vocab


def extract_test_features(
    events, sensor_vocab, activity_vocab, sensor_embeddings,
    window_size, stride, train_ratio=0.6, val_ratio=0.2, seed=42
):
    """Test features만 추출"""
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
    
    # Train/val/test split (동일한 seed 사용)
    np.random.seed(seed)
    np.random.shuffle(all_features)
    
    n_total = len(all_features)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Test features만 반환
    test_features = all_features[n_train + n_val:]
    
    return test_features


@torch.no_grad()
def evaluate_test(model, loader, criterion, device):
    """Test set 평가"""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    loss_components = {'L_cls': 0.0, 'L_move': 0.0, 'L_pos': 0.0, 'L_smooth': 0.0}
    
    for batch in loader:
        X_base = batch['X_base'].to(device)
        sensor_ids = batch['sensor_ids'].to(device)
        timestamps = batch['timestamps'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        logits, aux = model(X_base, sensor_ids, timestamps, return_aux=True)
        loss, losses = criterion(logits, labels, aux, model.pos_head.positions)
        
        # Accumulate
        total_loss += loss.item() * len(labels)
        for k in loss_components.keys():
            loss_components[k] += losses[k] * len(labels)
        
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
    
    return metrics, all_preds, all_labels


def main(args):
    print("=" * 80)
    print("🧪 Test Set Evaluation")
    print("=" * 80)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    # Load checkpoint
    print(f"\n📦 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    saved_args = checkpoint['args']
    sensor_vocab = checkpoint['sensor_vocab']
    activity_vocab = checkpoint['activity_vocab']
    
    print(f"   Trained at epoch: {checkpoint['epoch']}")
    print(f"   Val F1 (Macro): {checkpoint['val_f1_macro']:.2f}%")
    print(f"   Val Accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Load data
    print(f"\n📂 Loading events: {args.events_csv}")
    events = pd.read_csv(args.events_csv)
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    
    # Load embeddings
    print(f"\n📦 Loading embeddings: {args.embeddings}")
    emb_data = torch.load(args.embeddings, map_location='cpu', weights_only=True)
    sensor_embeddings = emb_data['embeddings'].numpy()
    
    # Extract test features
    print(f"\n🔧 Extracting test features...")
    test_features = extract_test_features(
        events, sensor_vocab, activity_vocab, sensor_embeddings,
        saved_args['window_size'],
        saved_args['stride'],
        saved_args['train_ratio'],
        saved_args['val_ratio'],
        saved_args['seed']
    )
    print(f"   Test samples: {len(test_features):,}")
    
    # Create test dataset
    test_ds = PVDataset(test_features, sensor_vocab, activity_vocab)
    test_loader = DataLoader(
        test_ds, batch_size=32,
        shuffle=False, collate_fn=collate_pv_features,
        num_workers=0
    )
    
    # Build model
    print(f"\n🏗️  Building model...")
    F_base = len(sensor_vocab) * 2 + 6 + sensor_embeddings.shape[1]
    
    model = SmartHomeModel(
        num_sensors=len(sensor_vocab),
        base_feat_dim=F_base,
        sensor_emb_dim=sensor_embeddings.shape[1],
        vel_dim=saved_args['vel_dim'],
        enc_hid=saved_args['hidden'],
        mmu_hid=saved_args['mmu_hid'],
        cmu_hid=saved_args['cmu_hid'],
        n_classes=len(activity_vocab)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Model loaded successfully!")
    
    # Loss
    criterion = MultiTaskLoss(
        lambda_move=saved_args['lambda_move'],
        lambda_pos=saved_args['lambda_pos'],
        lambda_smooth=saved_args['lambda_smooth']
    )
    
    # Evaluate
    print(f"\n🎯 Evaluating on test set...")
    test_metrics, preds, labels = evaluate_test(model, test_loader, criterion, device)
    
    # Results
    print("\n" + "=" * 80)
    print("📊 TEST SET RESULTS")
    print("=" * 80)
    print(f"\n✨ Performance:")
    print(f"   Accuracy: {test_metrics['acc']:.2f}%")
    print(f"   F1 Macro: {test_metrics['f1_macro']:.2f}%")
    print(f"   F1 Weighted: {test_metrics['f1_weighted']:.2f}%")
    print(f"\n📉 Loss:")
    print(f"   Total Loss: {test_metrics['loss']:.4f}")
    print(f"   L_cls: {test_metrics['L_cls']:.4f}")
    print(f"   L_move: {test_metrics['L_move']:.4f}")
    print(f"   L_pos: {test_metrics['L_pos']:.4f}")
    print(f"   L_smooth: {test_metrics['L_smooth']:.4f}")
    
    # Classification report
    inv_activity_vocab = {v: k for k, v in activity_vocab.items()}
    target_names = [inv_activity_vocab[i] for i in range(len(activity_vocab))]
    
    print("\n" + "=" * 80)
    print("📋 Classification Report:")
    print("=" * 80)
    print(classification_report(labels, preds, target_names=target_names, digits=4))
    
    # Confusion matrix
    print("\n" + "=" * 80)
    print("🔢 Confusion Matrix:")
    print("=" * 80)
    cm = confusion_matrix(labels, preds)
    print("     ", "  ".join(f"{name:>4s}" for name in target_names))
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>4s}", "  ".join(f"{val:>4d}" for val in row))
    
    # Save results
    test_results = {
        'test_acc': test_metrics['acc'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_weighted': test_metrics['f1_weighted'],
        'test_loss': test_metrics['loss'],
        'val_f1_macro': checkpoint['val_f1_macro'],
        'val_acc': checkpoint['val_acc'],
        'best_epoch': checkpoint['epoch'],
        'n_test': len(test_features),
        'loss_components': {
            'L_cls': test_metrics['L_cls'],
            'L_move': test_metrics['L_move'],
            'L_pos': test_metrics['L_pos'],
            'L_smooth': test_metrics['L_smooth']
        }
    }
    
    result_path = Path(args.checkpoint).with_suffix('.test_results.json')
    with open(result_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Test results saved to: {result_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Test Set")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--events-csv', type=str, required=True,
                        help='Path to events CSV')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Path to sensor embeddings')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    main(args)
