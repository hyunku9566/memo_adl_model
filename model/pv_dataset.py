#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position-Velocity Model용 Dataset Adapter
=========================================

RichFeatures를 Position-Velocity 모델 입력으로 변환:
- X_base: [T, F_base] - rich features (frame + ema + vel + emb)
- sensor_ids: [T] - 각 시점의 대표 센서 ID
- timestamps: [T] - 시간 (초 단위)

센서 ID 추출 전략:
1. 해당 시점에 활성화된 센서들 중 선택
2. 규칙: 가장 최근에 ON된 센서 (또는 가장 높은 EMA 값)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class PVDataset(Dataset):
    """
    Position-Velocity 모델용 Dataset
    
    RichFeatures를 입력으로 받아 (X_base, sensor_ids, timestamps, label) 반환
    """
    
    def __init__(
        self,
        features_list: List,  # List[RichFeatures]
        sensor_vocab: Dict[str, int],
        activity_vocab: Dict[str, int]
    ):
        """
        Args:
            features_list: RichFeatures 객체 리스트
            sensor_vocab: 센서명 → ID 매핑
            activity_vocab: 활동명 → ID 매핑
        """
        self.features_list = features_list
        self.sensor_vocab = sensor_vocab
        self.activity_vocab = activity_vocab
        self.num_sensors = len(sensor_vocab)
    
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
            - X_base: [T, F_base] - concatenated rich features
            - sensor_ids: [T] - 대표 센서 ID 시퀀스
            - timestamps: [T] - 시간 (초 단위)
            - label: scalar - 활동 레이블
            - length: scalar - 실제 시퀀스 길이
        """
        feat = self.features_list[idx]
        
        # 1) X_base 구성: [X_frame | X_ema | X_vel | X_emb]
        X_base = np.concatenate([
            feat.X_frame,   # [T, N_sensor]
            feat.X_ema,     # [T, N_sensor]
            feat.X_vel,     # [T, V]
            feat.X_emb      # [T, E]
        ], axis=-1)  # [T, F_base]
        
        # 2) sensor_ids 추출: 각 시점에서 가장 활성화된 센서 선택
        # 전략: X_ema 값이 가장 큰 센서 (EMA는 최근 활성도를 반영)
        sensor_ids = np.argmax(feat.X_ema, axis=-1)  # [T]
        
        # 3) timestamps 추출 (delta_t 행렬에서 첫 행 사용)
        # delta_t[0, :] = t[:] - t[0] → 상대 시간
        if feat.delta_t is not None and feat.delta_t.shape[0] > 0:
            timestamps = feat.delta_t[0, :]  # [T]
            # Normalize to reasonable range (avoid huge values)
            timestamps = timestamps / (timestamps.max() + 1e-6)  # [0, 1]
            timestamps = timestamps * 100  # scale to [0, 100] seconds (reasonable activity duration)
        else:
            # delta_t 없으면 균등 간격 가정 (1초 간격)
            timestamps = np.arange(feat.valid_length, dtype=np.float32)
        
        # 4) Convert to torch tensors
        return {
            'X_base': torch.from_numpy(X_base).float(),
            'sensor_ids': torch.from_numpy(sensor_ids).long(),
            'timestamps': torch.from_numpy(timestamps).float(),
            'label': torch.tensor(feat.label, dtype=torch.long),
            'length': torch.tensor(feat.valid_length, dtype=torch.long)
        }


def collate_pv_features(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Batch collation with padding
    
    Args:
        batch: List of dicts from PVDataset
    
    Returns:
        dict with keys:
        - X_base: [B, T_max, F_base]
        - sensor_ids: [B, T_max]
        - timestamps: [B, T_max]
        - labels: [B]
        - lengths: [B]
    """
    # 최대 길이
    max_len = max(item['length'].item() for item in batch)
    B = len(batch)
    F_base = batch[0]['X_base'].shape[-1]
    
    # Pad sequences
    X_base_pad = torch.zeros(B, max_len, F_base)
    ids_pad = torch.zeros(B, max_len, dtype=torch.long)
    ts_pad = torch.zeros(B, max_len)
    labels = torch.zeros(B, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    
    for i, item in enumerate(batch):
        L = item['length'].item()
        X_base_pad[i, :L] = item['X_base'][:L]
        ids_pad[i, :L] = item['sensor_ids'][:L]
        ts_pad[i, :L] = item['timestamps'][:L]
        labels[i] = item['label']
        lengths[i] = L
    
    return {
        'X_base': X_base_pad,
        'sensor_ids': ids_pad,
        'timestamps': ts_pad,
        'labels': labels,
        'lengths': lengths
    }


# ==================== Helper: RichFeatures → PVDataset ====================

def create_pv_datasets(
    train_features: List,
    val_features: List,
    sensor_vocab: Dict[str, int],
    activity_vocab: Dict[str, int]
) -> Tuple[PVDataset, PVDataset]:
    """
    RichFeatures 리스트로부터 train/val PVDataset 생성
    
    Args:
        train_features: List[RichFeatures] for training
        val_features: List[RichFeatures] for validation
        sensor_vocab: 센서 vocabulary
        activity_vocab: 활동 vocabulary
    
    Returns:
        train_dataset, val_dataset
    """
    train_ds = PVDataset(train_features, sensor_vocab, activity_vocab)
    val_ds = PVDataset(val_features, sensor_vocab, activity_vocab)
    return train_ds, val_ds


# ==================== Test Code ====================

if __name__ == "__main__":
    """간단한 동작 테스트"""
    from model.rich_features import RichFeatures
    
    # Dummy RichFeatures
    T = 100
    N_sensor = 30
    V = 6
    E = 32
    
    dummy_feat = RichFeatures(
        X_frame=np.random.rand(T, N_sensor).astype(np.float32),
        X_ema=np.random.rand(T, N_sensor).astype(np.float32),
        X_vel=np.random.rand(T, V).astype(np.float32),
        X_emb=np.random.rand(T, E).astype(np.float32),
        cond_feat=np.random.rand(T, 8).astype(np.float32),
        delta_t=np.cumsum(np.random.rand(T, T), axis=1).astype(np.float32),
        label=0,
        valid_length=T
    )
    
    sensor_vocab = {f'M{i:02d}': i for i in range(N_sensor)}
    activity_vocab = {'t1': 0, 't2': 1, 't3': 2, 't4': 3, 't5': 4}
    
    # Dataset
    ds = PVDataset([dummy_feat] * 4, sensor_vocab, activity_vocab)
    print(f"✓ Dataset length: {len(ds)}")
    
    # Get item
    item = ds[0]
    print(f"✓ X_base shape: {item['X_base'].shape}")  # [T, F_base]
    print(f"✓ sensor_ids shape: {item['sensor_ids'].shape}")  # [T]
    print(f"✓ timestamps shape: {item['timestamps'].shape}")  # [T]
    print(f"✓ label: {item['label']}")
    print(f"✓ length: {item['length']}")
    
    # Collate
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_pv_features)
    batch = next(iter(loader))
    print(f"\n✓ Batch X_base shape: {batch['X_base'].shape}")  # [B, T_max, F_base]
    print(f"✓ Batch sensor_ids shape: {batch['sensor_ids'].shape}")  # [B, T_max]
    print(f"✓ Batch timestamps shape: {batch['timestamps'].shape}")  # [B, T_max]
    print(f"✓ Batch labels shape: {batch['labels'].shape}")  # [B]
    print(f"✓ Batch lengths: {batch['lengths']}")  # [B]
