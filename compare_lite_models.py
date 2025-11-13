#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Lite Models
===================

Full vs Lite vs Minimal vs Ultra-Lite 비교 실험
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.train_pv_3split import (
    build_vocab,
    extract_features_3split,
    evaluate
)
from model.position_velocity_model import SmartHomeModel, MultiTaskLoss
from model.position_velocity_model_lite import (
    SmartHomeModelLite,
    SmartHomeModelMinimal,
    SmartHomeModelUltraLite,
    SmartHomeModelBaseline
)

import pandas as pd
from torch.utils.data import DataLoader
from model.pv_dataset import PVDataset, collate_pv_features


def count_parameters(model):
    """모델 파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_and_evaluate(
    model_class,
    model_name,
    args,
    events,
    sensor_vocab,
    activity_vocab,
    sensor_embeddings,
    device
):
    """모델 학습 및 평가"""
    
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    # Extract features
    print("Extracting features...")
    from train.train_pv_3split import extract_features_3split
    
    train_features, val_features, test_features = extract_features_3split(
        events, sensor_vocab, activity_vocab, sensor_embeddings,
        args.window_size, args.stride,
        args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    print(f"Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
    
    # Datasets
    train_ds = PVDataset(train_features, sensor_vocab, activity_vocab)
    val_ds = PVDataset(val_features, sensor_vocab, activity_vocab)
    test_ds = PVDataset(test_features, sensor_vocab, activity_vocab)
    
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        collate_fn=collate_pv_features, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        collate_fn=collate_pv_features, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        collate_fn=collate_pv_features, num_workers=0
    )
    
    # Build model
    F_base = len(sensor_vocab) * 2 + 6 + sensor_embeddings.shape[1]
    
    model = model_class(
        num_sensors=len(sensor_vocab),
        base_feat_dim=F_base,
        sensor_emb_dim=sensor_embeddings.shape[1],
        vel_dim=32,
        enc_hid=128,
        mmu_hid=128,
        cmu_hid=128,
        n_classes=len(activity_vocab)
    ).to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")
    
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
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    t_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
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
            
            train_loss += loss.item() * len(labels)
            pred = logits.argmax(dim=-1)
            train_correct += (pred == labels).sum().item()
            train_total += len(labels)
        
        train_acc = train_correct / train_total * 100
        train_loss = train_loss / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                X_base = batch['X_base'].to(device)
                sensor_ids = batch['sensor_ids'].to(device)
                timestamps = batch['timestamps'].to(device)
                labels = batch['labels'].to(device)
                
                logits, aux = model(X_base, sensor_ids, timestamps, return_aux=True)
                loss, losses = criterion(logits, labels, aux, model.pos_head.positions)
                
                val_loss += loss.item() * len(labels)
                pred = logits.argmax(dim=-1)
                val_correct += (pred == labels).sum().item()
                val_total += len(labels)
        
        val_acc = val_correct / val_total * 100
        val_loss = val_loss / val_total
        
        scheduler.step(val_acc)
        
        # Save best
        if val_acc > best_val_f1:
            best_val_f1 = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Acc={train_acc:.2f}% Val Acc={val_acc:.2f}%")
        
        history.append({
            'epoch': epoch,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch}")
            break
    
    train_time = time.time() - t_start
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            X_base = batch['X_base'].to(device)
            sensor_ids = batch['sensor_ids'].to(device)
            timestamps = batch['timestamps'].to(device)
            labels = batch['labels'].to(device)
            
            logits, aux = model(X_base, sensor_ids, timestamps, return_aux=True)
            pred = logits.argmax(dim=-1)
            test_correct += (pred == labels).sum().item()
            test_total += len(labels)
    
    test_acc = test_correct / test_total * 100
    
    results = {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_val_acc': best_val_f1,
        'test_acc': test_acc,
        'best_epoch': best_epoch,
        'train_time': train_time,
        'history': history
    }
    
    print(f"\nResults:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Best Val Acc: {best_val_f1:.2f}%")
    print(f"  Test Acc: {test_acc:.2f}%")
    print(f"  Training time: {train_time:.1f}s")
    
    return results


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    events = pd.read_csv(args.events_csv)
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    
    sensor_vocab, activity_vocab = build_vocab(events)
    
    # Load or synthesize sensor embeddings
    if getattr(args, 'no_skipgram', False):
        emb_dim = getattr(args, 'emb_dim', 32)
        num_sensors = len(sensor_vocab)
        print(f"Skipping skip-gram embeddings. Creating random embeddings: ({num_sensors}, {emb_dim})")
        sensor_embeddings = np.random.randn(num_sensors, emb_dim).astype(np.float32) * 0.1
    else:
        emb_data = torch.load(args.embeddings, map_location='cpu', weights_only=True)
        sensor_embeddings = emb_data['embeddings'].numpy()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Compare models
    # --models 파라미터로 학습할 모델 선택 가능
    available_models = {
        'full': (SmartHomeModel, "Full"),
        'lite': (SmartHomeModelLite, "Lite"),
        'minimal': (SmartHomeModelMinimal, "Minimal"),
        'ultra': (SmartHomeModelUltraLite, "Ultra-Lite"),
        'baseline': (SmartHomeModelBaseline, "Baseline")
    }
    
    # 학습할 모델 선택
    if args.models:
        selected_models = [m.lower() for m in args.models]
        models = [available_models[m] for m in selected_models if m in available_models]
    else:
        # 기본: 모든 모델 (Full 제외)
        models = [
            (SmartHomeModelLite, "Lite"),
            (SmartHomeModelMinimal, "Minimal"),
            (SmartHomeModelUltraLite, "Ultra-Lite"),
            (SmartHomeModelBaseline, "Baseline")
        ]
    
    print(f"\nTraining models: {[name for _, name in models]}\n")
    
    all_results = []
    
    for model_class, model_name in models:
        results = train_and_evaluate(
            model_class, model_name, args,
            events, sensor_vocab, activity_vocab,
            sensor_embeddings, device
        )
        all_results.append(results)
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<15} {'Params':>10} {'Val Acc':>10} {'Test Acc':>10} {'Time':>10}")
    print("-"*80)
    
    for res in all_results:
        print(f"{res['model_name']:<15} "
              f"{res['total_params']:>10,} "
              f"{res['best_val_acc']:>9.2f}% "
              f"{res['test_acc']:>9.2f}% "
              f"{res['train_time']:>9.1f}s")
    
    # Save results
    output_path = Path("comparison_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--events-csv', type=str, default='data/processed/events.csv')
    parser.add_argument('--embeddings', type=str, default='checkpoint/sensor_embeddings.pt')
    parser.add_argument('--window-size', type=int, default=100)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help='학습할 모델 선택: full, lite, minimal, ultra, baseline')
    parser.add_argument('--no-skipgram', action='store_true', help='Skip loading skip-gram embeddings and use random per-sensor embeddings')
    parser.add_argument('--emb-dim', type=int, default=32, help='Embedding dimension to use when --no-skipgram is set')
    
    args = parser.parse_args()
    main(args)
