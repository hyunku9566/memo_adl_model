"""Debug VelocityHead with real data"""
import torch
import numpy as np
from model.position_velocity_model import PositionHead, VelocityHead

# Load real data
data = np.load('checkpoint/sensor_embeddings.npz')
print(f"Available keys: {data.files}")

# Load events
import pandas as pd
events = pd.read_csv('data/processed/events.csv')
events['timestamp'] = pd.to_datetime(events['timestamp'])
events['timestamp_sec'] = (events['timestamp'] - events['timestamp'].iloc[0]).dt.total_seconds()

print(f"\nEvents shape: {events.shape}")
print(f"Timestamp (seconds) range: [{events['timestamp_sec'].min():.2f}, {events['timestamp_sec'].max():.2f}]")
print(f"Timestamp sample (first 10 seconds): {events['timestamp_sec'].head(10).tolist()}")

# Create a batch of real sensor_ids and timestamps
B, T = 4, 100
num_sensors = 30

# Sample real sensor_ids and timestamps
sensor_ids = torch.randint(0, num_sensors, (B, T))
# Use actual timestamp deltas from events
timestamp_diffs = events['timestamp_sec'].diff().fillna(0).values[:T]
timestamps = torch.tensor([np.cumsum(timestamp_diffs) for _ in range(B)], dtype=torch.float32)

print(f"\n=== Input ===")
print(f"sensor_ids shape: {sensor_ids.shape}")
print(f"timestamps shape: {timestamps.shape}")
print(f"timestamps[0, :10]: {timestamps[0, :10].tolist()}")
print(f"timestamp diffs[0, :10]: {(timestamps[0, 1:11] - timestamps[0, :10]).tolist()}")

# Test PositionHead
pos_head = PositionHead(num_sensors, init_scale=0.1)
pos = pos_head(sensor_ids)
print(f"\n=== PositionHead Output ===")
print(f"pos shape: {pos.shape}")
print(f"pos[0, :5]: {pos[0, :5]}")
print(f"pos min/max: [{pos.min():.4f}, {pos.max():.4f}]")
print(f"pos has NaN: {torch.isnan(pos).any().item()}")

# Test VelocityHead
vel_head = VelocityHead(d_model=32)
print(f"\n=== VelocityHead Forward ===")
try:
    vel, move_flag, aux = vel_head(pos, timestamps)
    print(f"✓ vel shape: {vel.shape}")
    print(f"✓ vel min/max: [{vel.min():.4f}, {vel.max():.4f}]")
    print(f"✓ vel has NaN: {torch.isnan(vel).any().item()}")
    
    print(f"\n=== Auxiliary Outputs ===")
    for k, v in aux.items():
        if isinstance(v, torch.Tensor):
            has_nan = torch.isnan(v).any().item()
            print(f"  {k:15s} shape={str(tuple(v.shape)):20s} range=[{v.min():.4f}, {v.max():.4f}]  NaN={has_nan}")
            if has_nan:
                print(f"    ⚠️  NaN locations: {torch.isnan(v).sum().item()} / {v.numel()}")
                # Find first NaN
                nan_mask = torch.isnan(v)
                if nan_mask.any():
                    indices = torch.where(nan_mask)
                    print(f"    First NaN at: batch={indices[0][0]}, time={indices[1][0]}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
