"""
Rich Feature Dataset for EMA Adaptive Decay Model
==================================================

RichFeatureExtractor로부터 생성된 Rich features를 PyTorch Dataset으로 래핑
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List

from model.rich_features import RichFeatures


class RichFeatureDataset(Dataset):
    """
    Rich features를 위한 PyTorch Dataset
    
    입력:
    - X_frame, X_ema, X_vel, X_emb를 concat한 X
    - cond_feat (λ 학습용)
    - delta_t (시간 차이)
    - lengths (유효 길이)
    - labels (활동 레이블)
    """
    
    def __init__(self, samples: List[RichFeatures]):
        """
        Args:
            samples: RichFeatures 리스트
        """
        self.samples = samples
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        # X_frame, X_ema, X_vel, X_emb를 concat
        # X_frame: (T, N_sensor)
        # X_ema: (T, N_sensor)
        # X_vel: (T, 6)
        # X_emb: (T, 32)
        X = np.concatenate([
            sample.X_frame,
            sample.X_ema,
            sample.X_vel,
            sample.X_emb
        ], axis=1)  # (T, F_in)
        
        return {
            'X': torch.from_numpy(X).float(),  # (T, F_in)
            'cond_feat': torch.from_numpy(sample.cond_feat).float(),  # (T, C)
            'delta_t': torch.from_numpy(sample.delta_t).float(),  # (T, T)
            'length': torch.tensor(sample.valid_length, dtype=torch.long),  # scalar
            'label': torch.tensor(sample.label, dtype=torch.long)  # scalar
        }


def collate_rich_features(batch):
    """
    배치 collate 함수
    
    Returns:
        dict with:
        - X: (B, T, F_in)
        - cond_feat: (B, T, C)
        - delta_t: (B, T, T)
        - lengths: (B,)
        - labels: (B,)
    """
    X = torch.stack([item['X'] for item in batch])
    cond_feat = torch.stack([item['cond_feat'] for item in batch])
    delta_t = torch.stack([item['delta_t'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'X': X,
        'cond_feat': cond_feat,
        'delta_t': delta_t,
        'lengths': lengths,
        'labels': labels
    }
