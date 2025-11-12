#!/usr/bin/env python3
"""
Debug script to identify NaN source in Position-Velocity model
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.rich_features import RichFeatureExtractor
from model.pv_dataset import PVDataset, collate_pv_features
from model.position_velocity_model import SmartHomeModel, MultiTaskLoss

# Load data
print("Loading data...")
events = pd.read_csv('data/processed/events.csv')
events['timestamp'] = pd.to_datetime(events['timestamp'])

sensors = sorted(events['sensor'].unique())
activities = sorted(events['activity'].dropna().unique())
sensor_vocab = {s: i for i, s in enumerate(sensors)}
activity_vocab = {a: i for i, a in enumerate(activities)}

# Load embeddings
emb_state = torch.load('checkpoint/sensor_embeddings_32d.pt', map_location='cpu')
sensor_embeddings = emb_state['embeddings'].numpy()

# Extract one small batch
extractor = RichFeatureExtractor(sensor_vocab, activity_vocab, sensor_embeddings)
activity = activities[0]
group = events[events['activity'] == activity].sort_values('timestamp').reset_index(drop=True)
features = extractor.extract_sequence(group, window_size=100, stride=50)

if len(features) == 0:
    print("No features extracted!")
    sys.exit(1)

# Create dataset
ds = PVDataset(features[:4], sensor_vocab, activity_vocab)
batch = collate_pv_features([ds[i] for i in range(len(ds))])

print(f"Batch shapes:")
print(f"  X_base: {batch['X_base'].shape}")
print(f"  sensor_ids: {batch['sensor_ids'].shape}")
print(f"  timestamps: {batch['timestamps'].shape}")

# Check for NaN in input
print(f"\nInput checks:")
print(f"  X_base NaN: {torch.isnan(batch['X_base']).any()}")
print(f"  timestamps NaN: {torch.isnan(batch['timestamps']).any()}")
print(f"  X_base range: [{batch['X_base'].min():.2f}, {batch['X_base'].max():.2f}]")
print(f"  timestamps range: [{batch['timestamps'].min():.2f}, {batch['timestamps'].max():.2f}]")

# Create model
F_base = len(sensor_vocab) * 2 + 6 + sensor_embeddings.shape[1]
model = SmartHomeModel(
    num_sensors=len(sensor_vocab),
    base_feat_dim=F_base,
    sensor_emb_dim=sensor_embeddings.shape[1],
    vel_dim=32,
    enc_hid=128,
    mmu_hid=128,
    cmu_hid=128,
    n_classes=len(activity_vocab)
)

print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Forward pass with checks
print("\nForward pass...")
X_base = batch['X_base']
sensor_ids = batch['sensor_ids']
timestamps = batch['timestamps']

try:
    # Step by step
    print("  1. PositionHead...")
    pos = model.pos_head(sensor_ids)
    print(f"     pos shape: {pos.shape}, NaN: {torch.isnan(pos).any()}, range: [{pos.min():.4f}, {pos.max():.4f}]")
    
    print("  2. VelocityHead...")
    vel, move_flag, aux_vel = model.vel_head(pos, timestamps)
    print(f"     vel shape: {vel.shape}, NaN: {torch.isnan(vel).any()}")
    print(f"     move_flag: {move_flag.shape}, NaN: {torch.isnan(move_flag).any()}")
    for k, v in aux_vel.items():
        if isinstance(v, torch.Tensor):
            has_nan = torch.isnan(v).any()
            print(f"       {k}: NaN={has_nan}, range=[{v.min():.4f}, {v.max():.4f}]")
    
    print("  3. MMU...")
    h_move = model.mmu(vel, move_flag)
    print(f"     h_move: {h_move.shape}, NaN: {torch.isnan(h_move).any()}")
    
    print("  4. CMU...")
    ctx_feat = torch.cat([X_base, vel], dim=-1)
    h_ctx = model.cmu(ctx_feat, move_flag)
    print(f"     h_ctx: {h_ctx.shape}, NaN: {torch.isnan(h_ctx).any()}")
    
    print("  5. Gate...")
    fused, gate_w, trig = model.gate(h_move, h_ctx, move_flag)
    print(f"     fused: {fused.shape}, NaN: {torch.isnan(fused).any()}")
    print(f"     gate_w: {gate_w.shape}, NaN: {torch.isnan(gate_w).any()}")
    
    print("  6. Encoder...")
    H = torch.cat([X_base, vel, fused], dim=-1)
    print(f"     H (input): {H.shape}, NaN: {torch.isnan(H).any()}")
    ctx, attn_w = model.encoder(H)
    print(f"     ctx: {ctx.shape}, NaN: {torch.isnan(ctx).any()}")
    
    print("  7. Classifier...")
    logits = model.classifier(ctx)
    print(f"     logits: {logits.shape}, NaN: {torch.isnan(logits).any()}")
    
    print("\n✓ Forward pass completed!")
    print(f"  Final logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    # Test loss
    print("\n8. Testing loss...")
    aux = {
        'pos': pos,
        'vel': vel,
        **aux_vel
    }
    
    criterion = MultiTaskLoss(lambda_move=0.5, lambda_pos=1.0, lambda_smooth=0.001)
    labels = batch['labels']
    
    loss, losses = criterion(logits, labels, aux, model.pos_head.positions)
    print(f"  Total loss: {loss.item()}")
    for k, v in losses.items():
        print(f"    {k}: {v}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
