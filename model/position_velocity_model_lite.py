#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position-Velocity Model - Lite Version
=======================================

TemporalEncoder를 경량화한 버전들을 제공합니다:

1. Lite Version: TCN 제거, GRU만 사용 (~150K params, -60%)
2. Minimal Version: GRU도 단순화 (~100K params, -75%)
3. Ultra-Lite: Attention만 사용 (~50K params, -87%)
4. Baseline: 시간 모델링 없이 선형 레이어만 사용 (~10K params, -97%)
"""

import torch
import torch.nn as nn
from typing import Tuple

# Import original components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from position_velocity_model import (
    PositionHead, VelocityHead, MMU, CMU, 
    GateAndTrigger, AdditiveAttention
)


# ==================== Lite TemporalEncoders ====================

class TemporalEncoderLite(nn.Module):
    """
    경량화 버전 1: TCN 제거, GRU만 사용
    
    파라미터: ~150K (원본의 40%)
    """
    
    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        
        # Projection
        self.proj = nn.Linear(in_dim, hid)
        
        # BiGRU only (no TCN)
        self.gru = nn.GRU(
            hid, hid, num_layers=1,
            batch_first=True, bidirectional=True
        )
        
        # Attention
        self.attn = AdditiveAttention(hid * 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)            # [B, T, hid]
        h, _ = self.gru(h)          # [B, T, 2*hid]
        ctx, w = self.attn(h)       # [B, 2*hid], [B, T]
        return ctx, w


class TemporalEncoderMinimal(nn.Module):
    """
    경량화 버전 2: 단방향 GRU + 더 작은 hidden
    
    파라미터: ~100K (원본의 25%)
    """
    
    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        
        # Projection to smaller dim
        small_hid = hid // 2  # 64
        self.proj = nn.Linear(in_dim, small_hid)
        
        # Unidirectional GRU (더 가벼움)
        self.gru = nn.GRU(
            small_hid, hid, num_layers=1,
            batch_first=True, bidirectional=False  # 단방향!
        )
        
        # Attention
        self.attn = AdditiveAttention(hid)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)            # [B, T, hid/2]
        h, _ = self.gru(h)          # [B, T, hid]
        ctx, w = self.attn(h)       # [B, hid], [B, T]
        
        # Output dimension 맞추기 (원본은 2*hid)
        ctx = torch.cat([ctx, ctx], dim=-1)  # [B, 2*hid]
        return ctx, w


class TemporalEncoderUltraLite(nn.Module):
    """
    최경량화 버전: Attention만 사용
    
    파라미터: ~50K (원본의 13%)
    """
    
    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        
        # Projection
        self.proj = nn.Linear(in_dim, hid)
        
        # Self-attention only
        self.attn = AdditiveAttention(hid)
        
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H = torch.relu(self.proj(X))  # [B, T, hid]
        
        # Self attention
        ctx, attn_w = self.attn(H)  # [B, hid*2], [B, T]
        
        return ctx, attn_w


class TemporalEncoderBaseline(nn.Module):
    """
    베이스라인: 시간 모델링 없이 선형 레이어만 사용
    
    단순히 평균 풀링 + MLP만 사용
    파라미터: ~10K (원본의 3%)
    """
    
    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        
        # Mean pooling + MLP
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid * 2)
        )
        
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simple mean pooling
        pooled = X.mean(dim=1)  # [B, in_dim]
        
        # MLP
        ctx = self.proj(pooled)  # [B, hid*2]
        
        # Dummy attention weights (uniform)
        B, T = X.shape[0], X.shape[1]
        attn_w = torch.ones(B, T, device=X.device) / T
        
        return ctx, attn_w


# ==================== Lite Models ====================

class SmartHomeModelLite(nn.Module):
    """Lite Version: TCN 제거"""
    
    def __init__(
        self,
        num_sensors: int,
        base_feat_dim: int,
        sensor_emb_dim: int = 32,
        vel_dim: int = 32,
        enc_hid: int = 128,
        mmu_hid: int = 128,
        cmu_hid: int = 128,
        n_classes: int = 5
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.base_feat_dim = base_feat_dim
        self.vel_dim = vel_dim
        
        # Original components
        self.pos_head = PositionHead(num_sensors, init_scale=0.1)
        self.vel_head = VelocityHead(d_model=vel_dim)
        
        self.cmu_in_dim = base_feat_dim + vel_dim
        self.mmu = MMU(in_dim=vel_dim, hid=mmu_hid)
        self.cmu = CMU(in_dim=self.cmu_in_dim, hid=cmu_hid)
        
        self.gate = GateAndTrigger(h_move=mmu_hid, h_ctx=cmu_hid)
        
        # Lite encoder
        self.encoder_in = base_feat_dim + vel_dim + max(mmu_hid, cmu_hid)
        self.encoder = TemporalEncoderLite(in_dim=self.encoder_in, hid=enc_hid)
        
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
        # Same as original
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


class SmartHomeModelMinimal(nn.Module):
    """Minimal Version: 단방향 GRU"""
    
    def __init__(
        self,
        num_sensors: int,
        base_feat_dim: int,
        sensor_emb_dim: int = 32,
        vel_dim: int = 32,
        enc_hid: int = 128,
        mmu_hid: int = 128,
        cmu_hid: int = 128,
        n_classes: int = 5
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.base_feat_dim = base_feat_dim
        self.vel_dim = vel_dim
        
        self.pos_head = PositionHead(num_sensors, init_scale=0.1)
        self.vel_head = VelocityHead(d_model=vel_dim)
        
        self.cmu_in_dim = base_feat_dim + vel_dim
        self.mmu = MMU(in_dim=vel_dim, hid=mmu_hid)
        self.cmu = CMU(in_dim=self.cmu_in_dim, hid=cmu_hid)
        
        self.gate = GateAndTrigger(h_move=mmu_hid, h_ctx=cmu_hid)
        
        # Minimal encoder
        self.encoder_in = base_feat_dim + vel_dim + max(mmu_hid, cmu_hid)
        self.encoder = TemporalEncoderMinimal(in_dim=self.encoder_in, hid=enc_hid)
        
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


class SmartHomeModelUltraLite(nn.Module):
    """Ultra-Lite Version: Attention만 사용"""
    
    def __init__(
        self,
        num_sensors: int,
        base_feat_dim: int,
        sensor_emb_dim: int = 32,
        vel_dim: int = 32,
        enc_hid: int = 128,
        mmu_hid: int = 128,
        cmu_hid: int = 128,
        n_classes: int = 5
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.base_feat_dim = base_feat_dim
        self.vel_dim = vel_dim
        
        self.pos_head = PositionHead(num_sensors, init_scale=0.1)
        self.vel_head = VelocityHead(d_model=vel_dim)
        
        self.cmu_in_dim = base_feat_dim + vel_dim
        self.mmu = MMU(in_dim=vel_dim, hid=mmu_hid)
        self.cmu = CMU(in_dim=self.cmu_in_dim, hid=cmu_hid)
        
        self.gate = GateAndTrigger(h_move=mmu_hid, h_ctx=cmu_hid)
        
        # Ultra-lite encoder
        self.encoder_in = base_feat_dim + vel_dim + max(mmu_hid, cmu_hid)
        self.encoder = TemporalEncoderUltraLite(in_dim=self.encoder_in, hid=enc_hid)
        
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


class SmartHomeModelBaseline(nn.Module):
    """
    베이스라인 모델: 시간 모델링 없이 단순 MLP만 사용
    
    TCN, GRU, Attention 없이 선형 레이어와 평균 풀링만 사용하여
    딥러닝 시간 모델링의 효과를 측정하기 위한 베이스라인
    """
    
    def __init__(
        self,
        num_sensors: int,
        base_feat_dim: int,
        sensor_emb_dim: int = 32,
        vel_dim: int = 32,
        enc_hid: int = 128,
        mmu_hid: int = 128,
        cmu_hid: int = 128,
        n_classes: int = 5
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.base_feat_dim = base_feat_dim
        self.vel_dim = vel_dim
        
        self.pos_head = PositionHead(num_sensors, init_scale=0.1)
        self.vel_head = VelocityHead(d_model=vel_dim)
        
        self.cmu_in_dim = base_feat_dim + vel_dim
        self.mmu = MMU(in_dim=vel_dim, hid=mmu_hid)
        self.cmu = CMU(in_dim=self.cmu_in_dim, hid=cmu_hid)
        
        self.gate = GateAndTrigger(h_move=mmu_hid, h_ctx=cmu_hid)
        
        # Baseline encoder (no temporal modeling)
        self.encoder_in = base_feat_dim + vel_dim + max(mmu_hid, cmu_hid)
        self.encoder = TemporalEncoderBaseline(in_dim=self.encoder_in, hid=enc_hid)
        
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


# ==================== Parameter Comparison ====================

if __name__ == "__main__":
    """파라미터 비교"""
    
    # Config
    num_sensors = 30
    base_feat_dim = 130  # 30*2 + 6 + 64
    
    # Original encoder
    from position_velocity_model import TemporalEncoder
    enc_full = TemporalEncoder(in_dim=base_feat_dim + 32 + 128, hid=128)
    params_full = sum(p.numel() for p in enc_full.parameters())
    
    # Lite versions
    enc_lite = TemporalEncoderLite(in_dim=base_feat_dim + 32 + 128, hid=128)
    params_lite = sum(p.numel() for p in enc_lite.parameters())
    
    enc_minimal = TemporalEncoderMinimal(in_dim=base_feat_dim + 32 + 128, hid=128)
    params_minimal = sum(p.numel() for p in enc_minimal.parameters())
    
    enc_ultra = TemporalEncoderUltraLite(in_dim=base_feat_dim + 32 + 128, hid=128)
    params_ultra = sum(p.numel() for p in enc_ultra.parameters())
    
    print("=" * 80)
    print("TemporalEncoder Parameter Comparison")
    print("=" * 80)
    print(f"Full Version:      {params_full:>8,} params (100%)")
    print(f"Lite Version:      {params_lite:>8,} params ({params_lite/params_full*100:>5.1f}%)")
    print(f"Minimal Version:   {params_minimal:>8,} params ({params_minimal/params_full*100:>5.1f}%)")
    print(f"Ultra-Lite Version:{params_ultra:>8,} params ({params_ultra/params_full*100:>5.1f}%)")
    print("=" * 80)
