#!/usr/bin/env python3
"""
Train EMA-Attention Based Adaptive Decay Memory Model
======================================================

전체 파이프라인:
1. Skip-gram 센서 임베딩 로드 (사전 학습)
2. Rich features 추출 (X_frame, X_ema, X_vel, X_emb)
3. EMA Adaptive Decay 모델 학습
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.rich_features import (
    RichFeatureExtractor,
    build_vocabulary,
    load_sensor_embeddings
)
from model.rich_dataset import RichFeatureDataset, collate_rich_features
from model.ema_adaptive_decay import create_ema_adaptive_decay_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train EMA Adaptive Decay Model")
    
    # Data
    parser.add_argument("--events-csv", type=Path, required=True,
                        help="Path to processed events.csv")
    parser.add_argument("--embeddings", type=Path, default=None,
                        help="Path to pre-trained sensor embeddings (optional)")
    
    # Feature extraction
    parser.add_argument("--window-size", type=int, default=100,
                        help="Sequence window size")
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding window stride (default: window_size)")
    parser.add_argument("--ema-alpha", type=float, default=0.6,
                        help="EMA smoothing factor")
    
    # Model architecture
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--tcn-layers", type=int, default=3,
                        help="Number of TCN layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--emb-dim", type=int, default=32,
                        help="Sensor embedding dimension")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Train split ratio")
    
    # System
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Output
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to save model checkpoint")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log interval (batches)")
    
    return parser.parse_args()


def set_seed(seed: int):
    """재현성을 위한 시드 고정"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int = 10
) -> Dict[str, float]:
    """한 에폭 학습"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # 데이터를 device로 이동
        X = batch['X'].to(device)
        cond_feat = batch['cond_feat'].to(device)
        delta_t = batch['delta_t'].to(device)
        lengths = batch['lengths'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        optimizer.zero_grad()
        logits, extras = model(X, cond_feat, delta_t, lengths)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{acc:.2f}%'
            })
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """검증"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            X = batch['X'].to(device)
            cond_feat = batch['cond_feat'].to(device)
            delta_t = batch['delta_t'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['labels'].to(device)
            
            logits, extras = model(X, cond_feat, delta_t, lengths)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Device 설정
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========== 1. 데이터 로드 ==========
    print(f"\n[1/5] Loading events from {args.events_csv}")
    events_df = pd.read_csv(args.events_csv)
    print(f"  - Total events: {len(events_df):,}")
    print(f"  - Columns: {events_df.columns.tolist()}")
    
    # Vocabulary 생성
    sensor_vocab, activity_vocab = build_vocabulary(events_df)
    num_sensors = len(sensor_vocab)
    num_classes = len(activity_vocab)
    print(f"  - Sensors: {num_sensors}, Activities: {num_classes}")
    print(f"  - Activity labels: {list(activity_vocab.keys())}")
    
    # ========== 2. Skip-gram 임베딩 로드 (옵션) ==========
    print(f"\n[2/5] Loading sensor embeddings")
    sensor_embeddings = None
    if args.embeddings and args.embeddings.exists():
        sensor_embeddings = load_sensor_embeddings(str(args.embeddings))
        print(f"  - Loaded embeddings: {sensor_embeddings.shape}")
    else:
        print("  - No pre-trained embeddings. Will use random initialization.")
        sensor_embeddings = np.random.randn(num_sensors, args.emb_dim).astype(np.float32) * 0.1
    
    # ========== 3. Rich features 추출 ==========
    print(f"\n[3/5] Extracting rich features")
    print(f"  - Window size: {args.window_size}")
    print(f"  - Stride: {args.stride or args.window_size}")
    print(f"  - EMA alpha: {args.ema_alpha}")
    
    extractor = RichFeatureExtractor(
        sensor_vocab=sensor_vocab,
        activity_vocab=activity_vocab,
        sensor_embeddings=sensor_embeddings,
        ema_alpha=args.ema_alpha,
        time_scale=1.0
    )
    
    samples = extractor.extract_sequence(
        events=events_df,
        window_size=args.window_size,
        stride=args.stride,
        pad=True
    )
    print(f"  - Total samples: {len(samples):,}")
    
    # Feature dimensions 확인
    sample0 = samples[0]
    print(f"  - X_frame shape: {sample0.X_frame.shape}")
    print(f"  - X_ema shape: {sample0.X_ema.shape}")
    print(f"  - X_vel shape: {sample0.X_vel.shape}")
    print(f"  - X_emb shape: {sample0.X_emb.shape}")
    print(f"  - cond_feat shape: {sample0.cond_feat.shape}")
    print(f"  - delta_t shape: {sample0.delta_t.shape}")
    
    # ========== 4. Dataset & DataLoader ==========
    print(f"\n[4/5] Creating datasets")
    dataset = RichFeatureDataset(samples)
    
    # Train/Val split
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"  - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_rich_features,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_rich_features,
        pin_memory=False
    )
    
    # ========== 5. 모델 생성 ==========
    print(f"\n[5/5] Creating model")
    model = create_ema_adaptive_decay_model(
        num_sensors=num_sensors,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        heads=args.heads,
        num_classes=num_classes,
        cond_dim=8,  # fixed
        tcn_layers=args.tcn_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # ========== 학습 루프 ==========
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, args.log_interval
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # History
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            
            # Convert Path objects to strings for checkpoint
            args_dict = vars(args).copy()
            for key, value in args_dict.items():
                if isinstance(value, Path):
                    args_dict[key] = str(value)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_acc': best_val_acc,
                'args': args_dict,
                'sensor_vocab': sensor_vocab,
                'activity_vocab': activity_vocab
            }
            
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, args.checkpoint)
            print(f"✓ Saved best model (Val Acc: {best_val_acc:.2f}%)")
    
    # ========== 결과 저장 ==========
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}\n")
    
    # Save training history
    metrics_path = args.checkpoint.with_suffix('.metrics.json')
    
    # Convert Path objects to strings for JSON serialization
    args_dict = vars(args).copy()
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
    
    metrics = {
        'history': history,
        'best_val_acc': best_val_acc,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'args': args_dict
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
