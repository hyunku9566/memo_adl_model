#!/usr/bin/env python3
"""
EMA-Attention Adaptive Decay Memory 모델 학습 스크립트

사용 방법:
python train/train_adaptive_decay_model.py \
  --events-csv data/processed/events.csv \
  --checkpoint checkpoint/adaptive_decay_model.pt \
  --window-size 100 \
  --batch-size 128 \
  --epochs 30 \
  --learning-rate 3e-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.data import load_events
from model.sequence_dataset import build_sequence_samples, SequenceSamples
from model.adaptive_decay_attention import (
    EMAAdaptiveDecayModel,
    AdaptiveDecayConfig,
    lengths_to_mask,
)
from utils.profiling import Timer


# ============================================================================
# 데이터셋 클래스
# ============================================================================
class AdaptiveDecayDataset(Dataset):
    """
    Adaptive Decay 모델용 데이터셋
    """
    def __init__(self, samples: SequenceSamples):
        self.samples = samples
        self.num_samples = len(samples.labels)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'sensor': (window_size,) int64,
                'state': (window_size,) int64,
                'value_type': (window_size,) int64,
                'numeric': (window_size,) float32,
                'numeric_mask': (window_size,) float32,
                'time': (window_size, 4) float32,
                'speed': (window_size,) float32,  # 조건 특징 (예시)
                'movement': (window_size,) float32,
                'label': () int64,
            }
        """
        sensor = torch.from_numpy(self.samples.sensor_seq[idx]).long()
        state = torch.from_numpy(self.samples.state_seq[idx]).long()
        value_type = torch.from_numpy(self.samples.value_type_seq[idx]).long()
        numeric = torch.from_numpy(self.samples.numeric_seq[idx]).float()
        numeric_mask = torch.from_numpy(self.samples.numeric_mask_seq[idx]).float()
        time_feats = torch.from_numpy(self.samples.time_features_seq[idx]).float()
        label = torch.tensor(self.samples.labels[idx], dtype=torch.long)

        # 조건 특징 생성 (speed, movement 등)
        # 간단한 예시: numeric 값의 변화를 "speed"로, 상태 변화를 "movement"로
        ws = self.samples.window_size
        speed = torch.zeros(ws, dtype=torch.float32)
        if ws > 1:
            numeric_diff = numeric[1:] - numeric[:-1]
            speed[1:] = torch.abs(numeric_diff).clamp(max=1.0)

        movement = (state[1:] != state[:-1]).float()
        movement = torch.cat([torch.zeros(1), movement])

        return {
            'sensor': sensor,
            'state': state,
            'value_type': value_type,
            'numeric': numeric,
            'numeric_mask': numeric_mask,
            'time': time_feats,
            'speed': speed,
            'movement': movement,
            'label': label,
        }


# ============================================================================
# 배치 콜레이터 (시간 차이 행렬 생성)
# ============================================================================
def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    배치 생성 시 delta_t 행렬 계산
    """
    # 기본 필드들
    keys = ['sensor', 'state', 'value_type', 'numeric', 'numeric_mask', 'time', 'speed', 'movement', 'label']
    
    batch_dict = {k: torch.stack([item[k] for item in batch]) for k in keys}
    
    B, T = batch_dict['sensor'].shape
    
    # 시간 차이 행렬: |t_query - t_key| (간단히 인덱스 차이 사용)
    time_indices = torch.arange(T, dtype=torch.float32)
    delta_t = torch.abs(time_indices.unsqueeze(0) - time_indices.unsqueeze(1))  # (T, T)
    delta_t = delta_t.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)
    
    batch_dict['delta_t'] = delta_t
    
    # 조건 특징 결합: [speed, movement, numeric_mask, time_of_day, day_of_week, ...]
    # 간단한 예시: speed와 movement만 사용
    cond_feat = torch.stack([
        batch_dict['speed'],      # (B, T)
        batch_dict['movement'],   # (B, T)
        batch_dict['numeric_mask'],  # (B, T)
        batch_dict['time'][:, :, 0],  # sin(time_of_day) -> (B, T)
        batch_dict['time'][:, :, 1],  # cos(time_of_day) -> (B, T)
        batch_dict['time'][:, :, 2],  # sin(day_of_week) -> (B, T)
        batch_dict['time'][:, :, 3],  # cos(day_of_week) -> (B, T)
        batch_dict['numeric']  # (B, T)
    ], dim=-1)  # (B, T, 8)
    
    batch_dict['cond_feat'] = cond_feat
    
    return batch_dict


# ============================================================================
# 입력 특징 생성 함수
# ============================================================================
def build_model_input(samples: SequenceSamples) -> torch.Tensor:
    """
    SequenceSamples로부터 모델 입력 X 생성
    
    X = [sensor_emb, state_emb, value_type_emb, numeric, numeric_mask, time_features]
    """
    num_samples, window_size = samples.sensor_seq.shape
    
    # 각 컴포넌트를 임베딩 또는 원본 그대로 사용
    num_sensors = len(samples.sensor_vocab)
    num_states = len(samples.state_vocab)
    num_value_types = len(samples.value_type_vocab)
    
    # One-hot 인코딩 사용 (또는 임베딩 레이어에서 처리)
    sensor_onehot = F.one_hot(
        torch.from_numpy(samples.sensor_seq), num_classes=num_sensors
    ).float()  # (N, T, num_sensors)
    
    state_onehot = F.one_hot(
        torch.from_numpy(samples.state_seq), num_classes=num_states
    ).float()  # (N, T, num_states)
    
    value_type_onehot = F.one_hot(
        torch.from_numpy(samples.value_type_seq), num_classes=num_value_types
    ).float()  # (N, T, num_value_types)
    
    numeric = torch.from_numpy(samples.numeric_seq).float().unsqueeze(-1)  # (N, T, 1)
    numeric_mask = torch.from_numpy(samples.numeric_mask_seq).float().unsqueeze(-1)  # (N, T, 1)
    time_feats = torch.from_numpy(samples.time_features_seq).float()  # (N, T, 4)
    
    # 결합
    X = torch.cat([
        sensor_onehot,
        state_onehot,
        value_type_onehot,
        numeric,
        numeric_mask,
        time_feats,
    ], dim=-1)  # (N, T, num_sensors + num_states + 4 + 1 + 1 + 4)
    
    return X


# ============================================================================
# 학습 함수
# ============================================================================
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    class_weights: torch.Tensor,
) -> Dict[str, float]:
    """한 epoch 학습"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # 배치 데이터 준비
        sensor = batch['sensor'].to(device).long()
        state = batch['state'].to(device).long()
        value_type = batch['value_type'].to(device).long()
        numeric = batch['numeric'].to(device).float()
        numeric_mask = batch['numeric_mask'].to(device).float()
        time_feats = batch['time'].to(device).float()
        labels = batch['label'].to(device).long()
        delta_t = batch['delta_t'].to(device).float()
        cond_feat = batch['cond_feat'].to(device).float()

        B, T = sensor.shape

        # Forward pass (임베딩 레이어 사용)
        logits, extras = model(
            sensor, state, value_type,
            numeric, numeric_mask, time_feats,
            cond_feat, delta_t, lengths=None
        )

        # Loss 계산
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 메트릭 계산
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += B

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_samples

    return {'loss': avg_loss, 'accuracy': avg_acc}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """모델 평가"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        sensor = batch['sensor'].to(device).long()
        state = batch['state'].to(device).long()
        value_type = batch['value_type'].to(device).long()
        numeric = batch['numeric'].to(device).float()
        numeric_mask = batch['numeric_mask'].to(device).float()
        time_feats = batch['time'].to(device).float()
        labels = batch['label'].to(device).long()
        delta_t = batch['delta_t'].to(device).float()
        cond_feat = batch['cond_feat'].to(device).float()

        # Forward pass (임베딩 레이어 사용)
        logits, _ = model(
            sensor, state, value_type,
            numeric, numeric_mask, time_feats,
            cond_feat, delta_t, lengths=None
        )
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    avg_acc = accuracy_score(all_labels, all_preds)
    avg_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return {'loss': avg_loss, 'accuracy': avg_acc, 'f1': avg_f1}


# ============================================================================
# 메인 학습 루프
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Train EMA-Attention Adaptive Decay Model")
    parser.add_argument("--events-csv", type=Path, default=Path("data/processed/events.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/adaptive_decay_model.pt"))
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-tcn-layers", type=int, default=3)
    parser.add_argument("--cond-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    
    # 시드 고정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ========== 데이터 로드 ==========
    print("Loading events...")
    with Timer("load_events") as t:
        events = load_events(args.events_csv)
    print(f"  Loaded {len(events.timestamps)} events in {t.stop():.2f}s")

    # ========== 시퀀스 샘플 생성 ==========
    print(f"Building sequence samples (window_size={args.window_size})...")
    with Timer("build_sequences") as t:
        samples = build_sequence_samples(events, window_size=args.window_size)
    print(f"  Generated {len(samples.labels)} samples in {t.stop():.2f}s")

    # ========== 학습/검증/테스트 분할 ==========
    num_samples = len(samples.labels)
    train_end = int(num_samples * 0.8)
    val_end = int(num_samples * 0.9)

    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, num_samples)

    print(f"Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    # ========== 어휘 크기 계산 (임베딩용) ==========
    num_sensors = int(samples.sensor_seq.max()) + 1
    num_states = int(samples.state_seq.max()) + 1
    num_value_types = int(samples.value_type_seq.max()) + 1
    
    print(f"Vocabulary sizes: sensor={num_sensors}, state={num_states}, value_type={num_value_types}")

    # ========== 데이터로더 ==========
    from torch.utils.data import Subset

    full_dataset = AdaptiveDecayDataset(samples)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    # ========== 모델 생성 ==========
    model = EMAAdaptiveDecayModel(
        num_sensors=num_sensors,
        num_states=num_states,
        num_value_types=num_value_types,
        num_classes=len(samples.label_names),
        sensor_embed_dim=64,
        state_embed_dim=16,
        value_type_embed_dim=8,
        hidden=args.hidden,
        heads=args.heads,
        num_tcn_layers=args.num_tcn_layers,
        cond_dim=args.cond_dim,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 손실함수, 옵티마이저 ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ========== 학습 루프 ==========
    best_val_acc = 0.0
    metrics_history = []

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    for epoch in range(args.epochs):
        # 학습
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, None)
        scheduler.step()

        # 평가
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
        }
        metrics_history.append(epoch_metrics)

        print(f"[{epoch+1:3d}/{args.epochs}] "
              f"train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} "
              f"val_f1={val_metrics['f1']:.4f}")

        # 최고 성능 모델 저장
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_sensors': num_sensors,
                'num_states': num_states,
                'num_value_types': num_value_types,
                'num_classes': len(samples.label_names),
                'hidden': args.hidden,
                'heads': args.heads,
                'sensor_vocab': samples.sensor_vocab,
                'state_vocab': samples.state_vocab,
                'value_type_vocab': samples.value_type_vocab,
                'label_names': samples.label_names,
            }, args.checkpoint)
            print(f"  -> Checkpoint saved to {args.checkpoint}")

    # ========== 테스트 평가 ==========
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted): {test_metrics['f1']:.4f}")

    # ========== 메트릭 저장 ==========
    metrics_file = args.checkpoint.parent / (args.checkpoint.stem + ".metrics.json")
    metrics_summary = {
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_metrics['accuracy']),
        'test_loss': float(test_metrics['loss']),
        'test_f1': float(test_metrics['f1']),
        'config': {
            'num_sensors': num_sensors,
            'num_states': num_states,
            'num_value_types': num_value_types,
            'window_size': args.window_size,
            'batch_size': args.batch_size,
            'hidden': args.hidden,
            'heads': args.heads,
            'num_tcn_layers': args.num_tcn_layers,
            'cond_dim': args.cond_dim,
            'dropout': args.dropout,
        },
        'history': metrics_history,
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
