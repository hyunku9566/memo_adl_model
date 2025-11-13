#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 평가 및 시각화 스크립트
"""

import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from model.position_velocity_model import SmartHomeModel
from model.rich_features import RichFeatureExtractor
from model.pv_dataset import PVDataset, collate_pv_features


def load_checkpoint_and_data(checkpoint_path, events_csv, embeddings_path):
    """체크포인트와 데이터 로드"""
    
    # 체크포인트 로드
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    sensor_vocab = ckpt['sensor_vocab']
    activity_vocab = ckpt['activity_vocab']
    saved_args = ckpt['args']
    
    # Convert dict to namespace if needed
    if isinstance(saved_args, dict):
        from argparse import Namespace
        args = Namespace(**saved_args)
    else:
        args = saved_args
    
    print(f"   Epoch: {ckpt['epoch']}")
    print(f"   Val Acc: {ckpt['val_acc']:.2f}%")
    print(f"   Val F1: {ckpt['val_f1_macro']:.2f}%")
    
    # 이벤트 로드
    print(f"\n📂 Loading events: {events_csv}")
    events = pd.read_csv(events_csv)
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    print(f"   Total events: {len(events):,}")
    
    # 임베딩 로드
    print(f"\n📦 Loading embeddings: {embeddings_path}")
    emb_data = torch.load(embeddings_path, map_location='cpu')
    sensor_embeddings = emb_data['embeddings'].numpy()
    print(f"   Embedding shape: {sensor_embeddings.shape}")
    
    # Feature extraction
    print(f"\n🔧 Extracting features (window={args.window_size}, stride={args.stride})")
    extractor = RichFeatureExtractor(
        sensor_vocab=sensor_vocab,
        activity_vocab=activity_vocab,
        sensor_embeddings=sensor_embeddings,
        ema_alpha=0.6,
        time_scale=1.0
    )
    
    all_features = []
    for activity, group in events.groupby('activity'):
        if pd.isna(activity):
            continue
        extractor.reset()  # 중요: 활동 간 상태 리셋
        group = group.sort_values('timestamp').reset_index(drop=True)
        features = extractor.extract_sequence(events=group, window_size=args.window_size, stride=args.stride)
        all_features.extend(features)
    
    # Train/val split (학습 시와 동일하게)
    np.random.seed(42)
    np.random.shuffle(all_features)
    split_idx = int(len(all_features) * args.train_ratio)
    train_features = all_features[:split_idx]
    val_features = all_features[split_idx:]
    
    print(f"   Train: {len(train_features)}, Val: {len(val_features)}")
    
    return ckpt, args, sensor_vocab, activity_vocab, sensor_embeddings, train_features, val_features


def build_model(ckpt, args, sensor_vocab, activity_vocab, sensor_embeddings, device):
    """모델 빌드 및 가중치 로드"""
    
    print(f"\n🏗️  Building model")
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
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    return model


def evaluate(model, val_features, sensor_vocab, activity_vocab, device, batch_size=32):
    """모델 평가"""
    
    print(f"\n🔍 Evaluating on validation set")
    val_dataset = PVDataset(val_features, sensor_vocab, activity_vocab)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pv_features)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            X_base = batch['X_base'].to(device)
            sensor_ids = batch['sensor_ids'].to(device)
            timestamps = batch['timestamps'].to(device)
            labels = batch['labels']
            
            logits = model(X_base, sensor_ids, timestamps)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, activities, save_path):
    """Confusion Matrix 시각화"""
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=activities, yticklabels=activities,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted Activity', fontsize=12)
    axes[0].set_ylabel('True Activity', fontsize=12)
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    
    # Normalized (%)
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=activities, yticklabels=activities,
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
    axes[1].set_xlabel('Predicted Activity', fontsize=12)
    axes[1].set_ylabel('True Activity', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized %)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")


def plot_training_curves(history_path, save_path):
    """학습 곡선 시각화"""
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(df['epoch'], df['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Accuracy Over Training', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Macro
    axes[0, 1].plot(df['epoch'], df['train_f1_macro'], 'b-', label='Train F1 Macro', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val_f1_macro'], 'r-', label='Val F1 Macro', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('F1 Macro (%)', fontsize=12)
    axes[0, 1].set_title('F1 Macro Score Over Training', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss Components
    axes[1, 0].plot(df['epoch'], df['train_L_cls'], label='L_cls', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['train_L_move'], label='L_move', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['train_L_pos'], label='L_pos', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['train_L_smooth'], label='L_smooth', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Training Loss Components', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Total Loss
    axes[1, 1].plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[1, 1].plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Total Loss', fontsize=12)
    axes[1, 1].set_title('Total Loss Over Training', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved: {save_path}")
    
    # Best performance
    best_f1_idx = df['val_f1_macro'].idxmax()
    best_acc_idx = df['val_acc'].idxmax()
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"\nBest Val F1 Macro: {df.loc[best_f1_idx, 'val_f1_macro']:.2f}% at Epoch {int(df.loc[best_f1_idx, 'epoch'])}")
    print(f"  - Val Acc: {df.loc[best_f1_idx, 'val_acc']:.2f}%")
    print(f"  - Train F1: {df.loc[best_f1_idx, 'train_f1_macro']:.2f}%")
    print(f"  - Train Acc: {df.loc[best_f1_idx, 'train_acc']:.2f}%")
    
    print(f"\nBest Val Accuracy: {df.loc[best_acc_idx, 'val_acc']:.2f}% at Epoch {int(df.loc[best_acc_idx, 'epoch'])}")
    print(f"  - Val F1: {df.loc[best_acc_idx, 'val_f1_macro']:.2f}%")
    
    final = df.iloc[-1]
    print(f"\nFinal Epoch ({int(final['epoch'])}):")
    print(f"  - Val F1: {final['val_f1_macro']:.2f}%")
    print(f"  - Val Acc: {final['val_acc']:.2f}%")


def plot_sensor_positions(model, sensor_vocab, save_path):
    """학습된 센서 위치 시각화"""
    
    device = next(model.parameters()).device
    idx_to_sensor = {idx: sensor for sensor, idx in sensor_vocab.items()}
    sensor_names = [idx_to_sensor[i] for i in range(len(sensor_vocab))]
    
    with torch.no_grad():
        dummy_sensors = torch.arange(len(sensor_vocab)).unsqueeze(0).to(device)
        positions = model.pos_head(dummy_sensors)
        positions = positions.squeeze(0).cpu().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    scatter = ax.scatter(positions[:, 0], positions[:, 1],
                        c=range(len(positions)), cmap='tab20',
                        s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for i, name in enumerate(sensor_names):
        ax.annotate(name, (positions[i, 0], positions[i, 1]),
                   fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    ax.set_xlabel('Position Dimension 1', fontsize=12)
    ax.set_ylabel('Position Dimension 2', fontsize=12)
    ax.set_title('Learned Sensor Positions (2D Embedding)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Sensor positions saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint/pv_model_final.pt')
    parser.add_argument('--events-csv', type=str, default='data/processed/events.csv')
    parser.add_argument('--embeddings', type=str, default='checkpoint/sensor_embeddings.pt')
    parser.add_argument('--history', type=str, default='checkpoint/pv_model_final.history.json')
    parser.add_argument('--output-dir', type=str, default='checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Load data
    ckpt, train_args, sensor_vocab, activity_vocab, sensor_embeddings, train_features, val_features = \
        load_checkpoint_and_data(args.checkpoint, args.events_csv, args.embeddings)
    
    # Build model
    model = build_model(ckpt, train_args, sensor_vocab, activity_vocab, sensor_embeddings, device)
    
    # Evaluate
    y_true, y_pred = evaluate(model, val_features, sensor_vocab, activity_vocab, device)
    
    # Classification report
    activities = sorted(activity_vocab.keys())
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=activities, digits=4))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, activities, f"{args.output_dir}/confusion_matrix_eval.png")
    
    # Plot training curves
    plot_training_curves(args.history, f"{args.output_dir}/training_curves.png")
    
    # Plot sensor positions
    plot_sensor_positions(model, sensor_vocab, f"{args.output_dir}/sensor_positions.png")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    print(f"  1. {args.output_dir}/training_curves.png")
    print(f"  2. {args.output_dir}/confusion_matrix_eval.png")
    print(f"  3. {args.output_dir}/sensor_positions.png")


if __name__ == '__main__':
    main()
