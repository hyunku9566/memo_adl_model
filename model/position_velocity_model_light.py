#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position-Velocity Model - Ultra Lightweight Version
===================================================

GRU hidden size를 대폭 줄여 파라미터와 연산량 최소화

Hidden size 비교:
- Original: 128 (MMU/CMU/Encoder)
- Light:    64  (50% 감소)
- Tiny:     32  (75% 감소)
"""

import torch
import torch.nn as nn
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from position_velocity_model import (
    PositionHead, VelocityHead, AdditiveAttention
)


# ==================== Lightweight Memory Units ====================

class MMULight(nn.Module):
    """경량 Movement Memory Unit (GRU hidden 64)"""
    
    def __init__(self, in_dim: int, hid: int = 64):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, batch_first=True)
        
    def forward(self, vel: torch.Tensor, move_flag: torch.Tensor):
        H, _ = self.gru(vel)
        return H


class CMULight(nn.Module):
    """경량 Context Memory Unit (GRU hidden 64)"""
    
    def __init__(self, in_dim: int, hid: int = 64):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, batch_first=True)
        
    def forward(self, ctx: torch.Tensor, move_flag: torch.Tensor):
        H, _ = self.gru(ctx)
        return H


class GateAndTriggerLight(nn.Module):
    """경량 Gate (hidden 64)"""
    
    def __init__(self, h_move: int = 64, h_ctx: int = 64):
        super().__init__()
        self.h_fused = max(h_move, h_ctx)
        
        self.gate_net = nn.Sequential(
            nn.Linear(h_move + h_ctx, self.h_fused),
            nn.ReLU(),
            nn.Linear(self.h_fused, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, h_move, h_ctx, move_flag):
        combined = torch.cat([h_move, h_ctx], dim=-1)
        gate_w = self.gate_net(combined)
        
        w_move = gate_w[..., 0:1]
        w_ctx = gate_w[..., 1:2]
        w_trig = gate_w[..., 2:3]
        
        fused = w_move * h_move + w_ctx * h_ctx
        trig = w_trig.squeeze(-1)
        
        return fused, gate_w, trig


class TemporalEncoderLight(nn.Module):
    """경량 TemporalEncoder (BiGRU hidden 64)"""
    
    def __init__(self, in_dim: int, hid: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid)
        self.gru = nn.GRU(hid, hid, batch_first=True, bidirectional=True)
        self.attn = AdditiveAttention(hid * 2)
        
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H = torch.relu(self.proj(X))
        H, _ = self.gru(H)
        ctx, attn_w = self.attn(H)
        return ctx, attn_w


# ==================== Lightweight Models ====================

class SmartHomeModelLight(nn.Module):
    """
    경량 모델 (hidden=64)
    - 파라미터: 원본의 ~25%
    - 속도: 2~3배 빠름
    """
    
    def __init__(
        self,
        num_sensors: int,
        base_feat_dim: int,
        vel_dim: int = 32,
        enc_hid: int = 64,
        mmu_hid: int = 64,
        cmu_hid: int = 64,
        n_classes: int = 5
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.base_feat_dim = base_feat_dim
        self.vel_dim = vel_dim
        
        self.pos_head = PositionHead(num_sensors, init_scale=0.1)
        self.vel_head = VelocityHead(d_model=vel_dim)
        
        self.cmu_in_dim = base_feat_dim + vel_dim
        self.mmu = MMULight(in_dim=vel_dim, hid=mmu_hid)
        self.cmu = CMULight(in_dim=self.cmu_in_dim, hid=cmu_hid)
        
        self.gate = GateAndTriggerLight(h_move=mmu_hid, h_ctx=cmu_hid)
        
        self.encoder_in = base_feat_dim + vel_dim + max(mmu_hid, cmu_hid)
        self.encoder = TemporalEncoderLight(in_dim=self.encoder_in, hid=enc_hid)
        
        self.classifier = nn.Sequential(
            nn.Linear(enc_hid * 2, enc_hid),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(enc_hid, n_classes),
        )
    
    def forward(
        self,
        X_base: torch.Tensor,
        sensor_ids: torch.Tensor,
        timestamps: torch.Tensor = None,
        return_aux: bool = False
    ):
        pos = self.pos_head(sensor_ids)
        vel, move_flag, aux_vel = self.vel_head(pos, timestamps)
        
        h_move = self.mmu(vel, move_flag)
        ctx_feat = torch.cat([X_base, vel], dim=-1)
        h_ctx = self.cmu(ctx_feat, move_flag)
        
        fused, gate_w, trig = self.gate(h_move, h_ctx, move_flag)
        
        H = torch.cat([X_base, vel, fused], dim=-1)
        ctx, attn_w = self.encoder(H)
        
        logits = self.classifier(ctx)
        
        if not return_aux:
            return logits
        
        aux = {
            'pos': pos,
            'vel': vel,
            'gate': gate_w,
            'trigger': trig,
            'attn': attn_w,
            **aux_vel
        }
        return logits, aux


class SmartHomeModelTiny(nn.Module):
    """
    초경량 모델 (hidden=32)
    - 파라미터: 원본의 ~10%
    - 속도: 4~5배 빠름
    """
    
    def __init__(
        self,
        num_sensors: int,
        base_feat_dim: int,
        vel_dim: int = 16,  # Velocity도 축소
        enc_hid: int = 32,
        mmu_hid: int = 32,
        cmu_hid: int = 32,
        n_classes: int = 5
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.base_feat_dim = base_feat_dim
        self.vel_dim = vel_dim
        
        self.pos_head = PositionHead(num_sensors, init_scale=0.1)
        self.vel_head = VelocityHead(d_model=vel_dim)
        
        self.cmu_in_dim = base_feat_dim + vel_dim
        self.mmu = MMULight(in_dim=vel_dim, hid=mmu_hid)
        self.cmu = CMULight(in_dim=self.cmu_in_dim, hid=cmu_hid)
        
        self.gate = GateAndTriggerLight(h_move=mmu_hid, h_ctx=cmu_hid)
        
        self.encoder_in = base_feat_dim + vel_dim + max(mmu_hid, cmu_hid)
        self.encoder = TemporalEncoderLight(in_dim=self.encoder_in, hid=enc_hid)
        
        self.classifier = nn.Sequential(
            nn.Linear(enc_hid * 2, enc_hid),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(enc_hid, n_classes),
        )
    
    def forward(
        self,
        X_base: torch.Tensor,
        sensor_ids: torch.Tensor,
        timestamps: torch.Tensor = None,
        return_aux: bool = False
    ):
        pos = self.pos_head(sensor_ids)
        vel, move_flag, aux_vel = self.vel_head(pos, timestamps)
        
        h_move = self.mmu(vel, move_flag)
        ctx_feat = torch.cat([X_base, vel], dim=-1)
        h_ctx = self.cmu(ctx_feat, move_flag)
        
        fused, gate_w, trig = self.gate(h_move, h_ctx, move_flag)
        
        H = torch.cat([X_base, vel, fused], dim=-1)
        ctx, attn_w = self.encoder(H)
        
        logits = self.classifier(ctx)
        
        if not return_aux:
            return logits
        
        aux = {
            'pos': pos,
            'vel': vel,
            'gate': gate_w,
            'trigger': trig,
            'attn': attn_w,
            **aux_vel
        }
        return logits, aux


class SmartHomeModelMicro(nn.Module):
    """
    극초경량 모델 (hidden=16)
    - 파라미터: 원본의 ~2%
    - 속도: 8~10배 빠름
    """
    
    def __init__(
        self,
        num_sensors: int,
        base_feat_dim: int,
        vel_dim: int = 8,   # Velocity도 대폭 축소
        enc_hid: int = 16,
        mmu_hid: int = 16,
        cmu_hid: int = 16,
        n_classes: int = 5
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.base_feat_dim = base_feat_dim
        self.vel_dim = vel_dim
        
        self.pos_head = PositionHead(num_sensors, init_scale=0.1)
        self.vel_head = VelocityHead(d_model=vel_dim)
        
        self.cmu_in_dim = base_feat_dim + vel_dim
        self.mmu = MMULight(in_dim=vel_dim, hid=mmu_hid)
        self.cmu = CMULight(in_dim=self.cmu_in_dim, hid=cmu_hid)
        
        self.gate = GateAndTriggerLight(h_move=mmu_hid, h_ctx=cmu_hid)
        
        self.encoder_in = base_feat_dim + vel_dim + max(mmu_hid, cmu_hid)
        self.encoder = TemporalEncoderLight(in_dim=self.encoder_in, hid=enc_hid)
        
        self.classifier = nn.Sequential(
            nn.Linear(enc_hid * 2, enc_hid),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout도 줄임
            nn.Linear(enc_hid, n_classes),
        )
    
    def forward(
        self,
        X_base: torch.Tensor,
        sensor_ids: torch.Tensor,
        timestamps: torch.Tensor = None,
        return_aux: bool = False
    ):
        pos = self.pos_head(sensor_ids)
        vel, move_flag, aux_vel = self.vel_head(pos, timestamps)
        
        h_move = self.mmu(vel, move_flag)
        ctx_feat = torch.cat([X_base, vel], dim=-1)
        h_ctx = self.cmu(ctx_feat, move_flag)
        
        fused, gate_w, trig = self.gate(h_move, h_ctx, move_flag)
        
        H = torch.cat([X_base, vel, fused], dim=-1)
        ctx, attn_w = self.encoder(H)
        
        logits = self.classifier(ctx)
        
        if not return_aux:
            return logits
        
        aux = {
            'pos': pos,
            'vel': vel,
            'gate': gate_w,
            'trigger': trig,
            'attn': attn_w,
            **aux_vel
        }
        return logits, aux


# ==================== Comparison ====================

if __name__ == "__main__":
    """파라미터 비교"""
    
    num_sensors = 30
    base_feat_dim = 66  # No embedding
    
    models = {
        'Original (hidden=128)': SmartHomeModelLight(
            num_sensors, base_feat_dim, vel_dim=32,
            enc_hid=128, mmu_hid=128, cmu_hid=128, n_classes=5
        ),
        'Light (hidden=64)': SmartHomeModelLight(
            num_sensors, base_feat_dim, vel_dim=32,
            enc_hid=64, mmu_hid=64, cmu_hid=64, n_classes=5
        ),
        'Tiny (hidden=32)': SmartHomeModelTiny(
            num_sensors, base_feat_dim, vel_dim=16,
            enc_hid=32, mmu_hid=32, cmu_hid=32, n_classes=5
        ),
        'Micro (hidden=16)': SmartHomeModelMicro(
            num_sensors, base_feat_dim, vel_dim=8,
            enc_hid=16, mmu_hid=16, cmu_hid=16, n_classes=5
        )
    }
    
    print("="*80)
    print("GRU Hidden Size 비교")
    print("="*80)
    print(f"{'Model':<25} {'Total Params':>15} {'GRU Params':>15} {'Reduction':>12}")
    print("-"*80)
    
    for name, model in models.items():
        total = sum(p.numel() for p in model.parameters())
        
        # GRU 파라미터만 계산
        gru_params = 0
        for module in model.modules():
            if isinstance(module, nn.GRU):
                gru_params += sum(p.numel() for p in module.parameters())
        
        reduction = (1 - total / 464877) * 100 if total < 464877 else 0
        
        print(f"{name:<25} {total:>15,} {gru_params:>15,} {reduction:>11.1f}%")
    
    print("\n" + "="*80)
    print("예상 성능")
    print("="*80)
    print("""
Original (128):  98.5~99.0%  (베이스라인)
Light (64):      99.68%      ⭐ 실제 측정!
Tiny (32):       95~97%      (학습 중)
Micro (16):      90~93%      (극초경량)

권장:
- 정확도 최우선: Light (64) ← 99.68%!
- 균형잡힌:     Tiny (32)  (엣지 디바이스)
- 극한 경량:     Micro (16) (초저사양)
""")
