"""Debug real training batch"""
import torch
import pandas as pd
import sys
sys.path.insert(0, '/home/lee/memo_model_adl')

from train.train_pv_model import extract_features, build_vocab
from model.pv_dataset import collate_pv_features, create_pv_datasets

# Load data
events = pd.read_csv('data/processed/events.csv')
events['timestamp'] = pd.to_datetime(events['timestamp'])
print(f"Total events: {len(events)}")

# Vocabularies
sensor_vocab, activity_vocab = build_vocab(events)
print(f"Sensors: {len(sensor_vocab)}, Activities: {len(activity_vocab)}")

# Load embeddings
emb_data = torch.load('checkpoint/sensor_embeddings.pt', map_location='cpu')
sensor_emb = emb_data['embeddings'].numpy()
print(f"Embeddings shape: {sensor_emb.shape}")

# Extract features
train_features, val_features = extract_features(
    events, sensor_vocab, activity_vocab, sensor_emb,
    window_size=100, stride=10, train_ratio=0.8
)
print(f"Train features: {len(train_features)}")
print(f"Val features: {len(val_features)}")

# Create datasets
train_ds, val_ds = create_pv_datasets(train_features, val_features, sensor_vocab, activity_vocab)

# Get first batch
from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=collate_pv_features)
batch = next(iter(train_loader))
X_base = batch['X_base']
sensor_ids = batch['sensor_ids']
timestamps = batch['timestamps']
labels = batch['labels']

print(f"\n=== Batch Info ===")
print(f"X_base shape: {X_base.shape}")
print(f"sensor_ids shape: {sensor_ids.shape}")
print(f"timestamps shape: {timestamps.shape}")
print(f"labels shape: {labels.shape}")

print(f"\n=== Timestamps Analysis ===")
print(f"timestamps min: {timestamps.min().item():.4f}")
print(f"timestamps max: {timestamps.max().item():.4f}")
print(f"timestamps[0, :10]: {timestamps[0, :10].tolist()}")

# Compute dt
dt_raw = timestamps[:, 1:] - timestamps[:, :-1]
print(f"\n=== Delta-t Analysis ===")
print(f"dt min: {dt_raw.min().item():.4f}")
print(f"dt max: {dt_raw.max().item():.4f}")
print(f"dt mean: {dt_raw.mean().item():.4f}")
print(f"dt std: {dt_raw.std().item():.4f}")

# Check for problematic values
dt_zero = (dt_raw == 0).sum().item()
dt_tiny = (dt_raw < 0.01).sum().item()
dt_huge = (dt_raw > 1000).sum().item()
print(f"\ndt == 0: {dt_zero} / {dt_raw.numel()}")
print(f"dt < 0.01: {dt_tiny} / {dt_raw.numel()}")
print(f"dt > 1000: {dt_huge} / {dt_raw.numel()}")

# Test forward pass
from model.position_velocity_model import SmartHomeModel, MultiTaskLoss
import torch.nn as nn

model = SmartHomeModel(
    num_sensors=30,
    base_feat_dim=130,
    sensor_emb_dim=64,
    vel_dim=32,
    enc_hid=128,
    mmu_hid=128,
    cmu_hid=128,
    n_classes=5
)

print(f"\n=== Forward Pass ===")
try:
    logits, aux = model(X_base, sensor_ids, timestamps, return_aux=True)
    
    print(f"✓ logits shape: {logits.shape}")
    print(f"✓ logits has NaN: {torch.isnan(logits).any().item()}")
    print(f"✓ logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    for k, v in aux.items():
        if isinstance(v, torch.Tensor):
            has_nan = torch.isnan(v).any().item()
            print(f"  {k:15s} NaN={has_nan}  range=[{v.min():.4f}, {v.max():.4f}]")
    
    # Test backward
    print(f"\n=== Backward Pass ===")
    criterion = MultiTaskLoss(lambda_move=0.5, lambda_pos=0.1, lambda_smooth=0.01)
    loss, losses = criterion(logits, labels, aux, model.pos_head.positions)
    
    print(f"✓ loss: {loss.item():.4f}")
    print(f"✓ losses: {losses}")
    print(f"✓ loss has NaN: {torch.isnan(loss).any().item()}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    print(f"\n=== Gradients ===")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_nan = torch.isnan(param.grad).any().item()
            grad_inf = torch.isinf(param.grad).any().item()
            if grad_nan or grad_inf:
                print(f"  ⚠️  {name:40s} NaN={grad_nan} Inf={grad_inf}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
